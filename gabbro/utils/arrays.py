import copy

import awkward as ak
import numpy as np
import torch
import vector

from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)

vector.register_awkward()


def signed_log(x, eps=1e-5):
    """Signed logarithm function that avoids infinities for zero values.

    Parameters
    ----------
    x : np.ndarray or ak.Array
        Input value.
    eps : float, optional
        Small epsilon to avoid infinities. Default is 1e-5.
    """
    return np.sign(x) * np.log1p(np.abs(x) + eps)


def signed_exp(x, eps=1e-5):
    """Signed exponential function that avoids infinities for zero values.

    Parameters
    ----------
    x : np.ndarray or ak.Array
        Input value.
    eps : float, optional
        Small epsilon to undo the effect of signed_log. Default is 1e-5.
    """
    return np.sign(x) * (np.expm1(np.abs(x)) - eps)


def get_causal_mask(x: torch.Tensor, fill_value=float("-inf")) -> torch.Tensor:
    """Create a causal mask for a sequence of length `seq_len`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, feats).
    fill_value : any, optional
        Value to fill the upper triangle of the mask with. Default is float("-inf").
    """
    seq_len = x.size(1)
    return torch.triu(torch.full((seq_len, seq_len), fill_value), diagonal=1)


def p4s_from_ptetaphimass(
    ak_arr,
    field_name_pt="part_pt",
    field_name_eta="part_etarel",
    field_name_phi="part_phirel",
    field_name_mass="part_mass",
):
    """Create a Momentum4D array from pt, eta, phi, mass fields in an awkward array.

    Parameters
    ----------
    ak_arr : ak.Array
        Array with fields part_pt, part_etarel, part_phirel, part_mass.
    field_name_pt : str, optional
        Name of the field containing the transverse momentum, by default "part_pt".
    field_name_eta : str, optional
        Name of the field containing the pseudorapidity, by default "part_etarel".
    field_name_phi : str, optional
        Name of the field containing the azimuthal angle, by default "part_phirel".
    field_name_mass : str, optional
        Name of the field containing the mass, by default
    """
    return ak.zip(
        {
            "pt": ak_arr[field_name_pt],
            "eta": ak_arr[field_name_eta],
            "phi": ak_arr[field_name_phi],
            "mass": ak_arr[field_name_mass]
            if field_name_mass in ak_arr.fields
            else ak.zeros_like(ak_arr[field_name_pt]),
        },
        with_name="Momentum4D",
    )


def ak_pad(x: ak.Array, maxlen: int, axis: int = 1, fill_value=0, return_mask=False):
    """Function to pad an awkward array to a specified length. The array is padded along the
    specified axis.

    Parameters
    ----------
    x : awkward array
        Array to pad.
    maxlen : int
        Length to pad to.
    axis : int, optional
        Axis along which to pad. Default is 1.
    fill_value : float or int, optional
        Value to use for padding. Default is 0.
    return_mask : bool, optional
        If True, also return a mask array indicating which values are padded.
        Default is False.
        If the input array has fields, the mask is created from the first field.

    Returns
    -------
    awkward array
        Padded array.
    mask : awkward array
        Mask array indicating which values are padded. Only returned if return_mask is True.
    """
    padded_x = ak.fill_none(ak.pad_none(x, maxlen, axis=axis, clip=True), fill_value)
    if return_mask:
        if len(x.fields) >= 1:
            mask = ak.ones_like(x[x.fields[0]], dtype="bool")
        else:
            mask = ak.ones_like(x, dtype="bool")
        mask = ak.fill_none(ak.pad_none(mask, maxlen, axis=axis, clip=True), False)
        return padded_x, mask
    return padded_x


def combine_ak_arrays(*arrays):
    """Function to combine multiple awkward arrays. The arrays should have different fields.

    Parameters
    ----------
    *arrays : ak.Array
        Input arrays to combine.

    Returns
    -------
    ak.Array
        Combined array.
    """
    combined_fields = {}
    for arr in arrays:
        if arr is None:
            continue
        if set(combined_fields.keys()) & set(arr.fields):
            dict_with_fields = {f"arr{i}": arr.fields for i, arr in enumerate(arrays)}
            raise ValueError(
                "You are trying to merge multiple ak.Arrays but they have common field names. "
                f"The common field names are: {set(combined_fields.keys()) & set(arr.fields)} "
                f"The individual field names are: {dict_with_fields}"
            )
        combined_fields.update({field: arr[field] for field in arr.fields})
    return ak.Array(combined_fields)


def np_to_ak(x: np.ndarray, names: list, mask: np.ndarray = None, dtype="float32"):
    """Function to convert a numpy array and its mask to an awkward array. The features
    corresponding to the names are assumed to correspond to the last axis of the array.

    Parameters
    ----------
    x : np.ndarray
        Array to convert.
    names : list
        List of field names (corresponding to the features in x along the last dimension).
    mask : np.ndarray, optional
        Mask array. Default is None. If x is an array of shape (n, m, k), the mask should
        be of shape (n, m).
    dtype : str, optional
        Data type to convert the values to. Default is "float32".
    """

    if mask is None:
        mask = np.ones_like(x[..., 0], dtype="bool")

    return ak.Array(
        {
            name: ak.values_astype(
                ak.drop_none(ak.mask(ak.Array(x[..., i]), mask != 0)),
                dtype,
            )
            for i, name in enumerate(names)
        }
    )


def ak_to_np_stack(ak_array: ak.Array, names: list = None, axis: int = -1):
    """Function to convert an awkward array to a numpy array by stacking the values of the
    specified fields. This is much faster than ak.to_numpy(ak_array) for large arrays.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    names : list, optional
        List of field names to convert. Default is None.
    axis : int, optional
        Axis along which to stack the values. Default is -1.
    """
    if names is None:
        raise ValueError("names must be specified")
    return ak.to_numpy(
        np.stack(
            [ak.to_numpy(ak.values_astype(ak_array[name], "float32")) for name in names],
            axis=axis,
        )
    )


def np_PtEtaPhi_to_Momentum4D(arr, mask, log_pt=False):
    """Convert numpy array with 4-momenta to ak array of Momentum4D objects.
    NOTE: the input array is assumed to be in (pT, eta, phi) format, thus mass = 0.

    Expects an array of shape (batch_size, num_particles, 3)
    where the last dimension is (pt, eta, phi)

    Returns an ak array of shape (batch_size, var, 4) of Momentum4D objects

    If log_pt is True, the corresponding variable is exponentiated
    before being passed to Momentum4D

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (batch_size, num_particles, 3)
    mask : np.ndarray
        Mask array of shape (batch_size, num_particles)
    log_pt : bool, optional
        Whether to exponentiate pt, by default False

    Returns
    -------
    ak.Array
        Array of Momentum4D objects
    """

    p4 = ak.zip(
        {
            "pt": np.clip(arr[:, :, 0], 0, None) if not log_pt else np.exp(arr[:, :, 0]),
            "eta": arr[:, :, 1],
            "phi": arr[:, :, 2],
            "mass": ak.zeros_like(arr[:, :, 0]),
        },
        with_name="Momentum4D",
    )
    # mask the array
    ak_mask = ak.Array(mask)
    return ak.drop_none(ak.mask(p4, ak_mask == 1))


def ak_select_and_preprocess(
    ak_array: ak.Array,
    pp_dict=None,
    inverse=False,
    suppress_warnings=False,
):
    """Function to select and pre-process fields from an awkward array.

    Parameters
    ----------
    ak_array : awkward array
        Array to convert.
    pp_dict : dict, optional
        Dictionary with pre-processing values for each field. Default is None.
        The dictionary should have the following format:
        {
            "field_name_1": {
                "multiply_by": 1,
                "subtract_by": 0,
                "func": "np.log",
                "inv_func": "np.exp",
                "clip_min_input_space": -5.0,  # Clip in original input space
                "clip_max_input_space": 5.0,
                "clip_min_preproc_space": 0.0,  # Clip after all transformations
                "clip_max_preproc_space": 10.0,
                # Legacy (deprecated):
                "clip_min": 0.0,  # Will be treated as clip_min_preproc_space
                "clip_max": 10.0   # Will be treated as clip_max_preproc_space
            },
            "field_name_2": {"multiply_by": 1, "subtract_by": 0, "func": None},
            ...
        }

        Available parameters for each field:
        - multiply_by: float, scaling factor applied at the end
        - subtract_by: float, offset applied before scaling
        - func: str, function to apply (e.g., "np.log", "signed_log")
        - inv_func: str, inverse function for the func
        - clip_min_input_space: float, minimum value to clip in input space
        - clip_max_input_space: float, maximum value to clip in input space
        - clip_min_preproc_space: float, minimum value to clip in preprocessed space
        - clip_max_preproc_space: float, maximum value to clip in preprocessed space
        - clip_min: float, legacy parameter (treated as clip_min_preproc_space)
        - clip_max: float, legacy parameter (treated as clip_max_preproc_space)
        - larger_than: float, selection cut (keep values > this)
        - smaller_than: float, selection cut (keep values < this)
        - bin_edges: array, bin edges for discretization
        - binning: tuple, (start, stop, n_bins) for automatic bin edge creation

    inverse : bool, optional
        If True, the inverse of the pre-processing is applied. Default is False.
    suppress_warnings : bool, optional
        If True, suppress warnings about clipping and non-invertible operations.
    """
    if pp_dict is None:
        pp_dict = {}
    else:
        pp_dict = copy.deepcopy(pp_dict)

    # define initial mask as all True
    first_feat = list(pp_dict.keys())[0]
    selection_mask = ak.ones_like(ak_array[first_feat], dtype="bool")

    for name, params in pp_dict.items():
        if params is None:
            pp_dict[name] = {
                "subtract_by": 0,
                "multiply_by": 1,
                "func": None,
                "inv_func": None,
                "larger_than": None,
                "smaller_than": None,
                "binning": None,
                "bin_edges": None,
                "clip_min": None,
                "clip_max": None,
                "clip_min_input_space": None,
                "clip_max_input_space": None,
                "clip_min_preproc_space": None,
                "clip_max_preproc_space": None,
            }
        else:
            if "subtract_by" not in params:
                pp_dict[name]["subtract_by"] = 0
            if "multiply_by" not in params:
                pp_dict[name]["multiply_by"] = 1
            if "func" not in params:
                pp_dict[name]["func"] = None
            if "inv_func" not in params:
                pp_dict[name]["inv_func"] = None
            if "larger_than" not in params:
                pp_dict[name]["larger_than"] = None
            if "smaller_than" not in params:
                pp_dict[name]["smaller_than"] = None
            if "bin_edges" not in params:
                pp_dict[name]["bin_edges"] = None
            if "binning" not in params:
                pp_dict[name]["binning"] = None
            elif pp_dict[name]["binning"] is not None:
                # convert tuple of (start, end, n_bins) to np.linspace
                start, stop, n_bins = pp_dict[name]["binning"]
                if pp_dict[name].get("bin_edges") is None:
                    pp_dict[name]["bin_edges"] = np.linspace(start, stop, int(n_bins))
                print(
                    f"Applying binning to field {name} with np.linspace({start}, {stop}, {n_bins})"
                )
            if "clip_min" not in params:
                pp_dict[name]["clip_min"] = None
            if "clip_max" not in params:
                pp_dict[name]["clip_max"] = None
            if "clip_min_input_space" not in params:
                pp_dict[name]["clip_min_input_space"] = None
            if "clip_max_input_space" not in params:
                pp_dict[name]["clip_max_input_space"] = None
            if "clip_min_preproc_space" not in params:
                pp_dict[name]["clip_min_preproc_space"] = None
            if "clip_max_preproc_space" not in params:
                pp_dict[name]["clip_max_preproc_space"] = None

            # Handle legacy clip_min/clip_max by translating to preprocessed space
            if pp_dict[name]["clip_min"] is not None:
                if pp_dict[name]["clip_min_preproc_space"] is not None:
                    raise ValueError(
                        f"Cannot specify both 'clip_min' and 'clip_min_preproc_space' for field {name}. "
                        "Use 'clip_min_preproc_space' for clipping in preprocessed space or "
                        "'clip_min_input_space' for clipping in input space."
                    )
                if not suppress_warnings:
                    logger.warning(
                        f"'clip_min' is deprecated for field '{name}'. "
                        "Use 'clip_min_preproc_space' for preprocessed space clipping or "
                        "'clip_min_input_space' for input space clipping."
                    )
                pp_dict[name]["clip_min_preproc_space"] = pp_dict[name]["clip_min"]

            if pp_dict[name]["clip_max"] is not None:
                if pp_dict[name]["clip_max_preproc_space"] is not None:
                    raise ValueError(
                        f"Cannot specify both 'clip_max' and 'clip_max_preproc_space' for field {name}. "
                        "Use 'clip_max_preproc_space' for clipping in preprocessed space or "
                        "'clip_max_input_space' for clipping in input space."
                    )
                if not suppress_warnings:
                    logger.warning(
                        f"'clip_max' is deprecated for field '{name}'. "
                        "Use 'clip_max_preproc_space' for preprocessed space clipping or "
                        "'clip_max_input_space' for input space clipping."
                    )
                pp_dict[name]["clip_max_preproc_space"] = pp_dict[name]["clip_max"]

            # Validation for input space clipping
            if (
                pp_dict[name]["clip_min_input_space"] is not None
                or pp_dict[name]["clip_max_input_space"] is not None
            ):
                if not suppress_warnings:
                    logger.warning(
                        f"You are clipping the values of the feature '{name}' in input space with "
                        f"clip_min_input_space = {pp_dict[name]['clip_min_input_space']} and "
                        f"clip_max_input_space = {pp_dict[name]['clip_max_input_space']}. "
                        "Make sure this is intended. THIS IS NOT INVERTIBLE."
                    )

            # Validation for preprocessed space clipping
            if (
                pp_dict[name]["clip_min_preproc_space"] is not None
                or pp_dict[name]["clip_max_preproc_space"] is not None
            ):
                if not suppress_warnings:
                    logger.warning(
                        f"You are clipping the values of the feature '{name}' in preprocessed space with "
                        f"clip_min_preproc_space = {pp_dict[name]['clip_min_preproc_space']} and "
                        f"clip_max_preproc_space = {pp_dict[name]['clip_max_preproc_space']}. "
                        "Make sure this is intended. THIS IS NOT INVERTIBLE."
                    )

            # Validation for input space clip ranges
            if (
                pp_dict[name]["clip_min_input_space"] is not None
                and pp_dict[name]["clip_max_input_space"] is not None
            ):
                if pp_dict[name]["clip_min_input_space"] > pp_dict[name]["clip_max_input_space"]:
                    raise ValueError(
                        "clip_min_input_space must be smaller than clip_max_input_space. "
                        f"You have clip_min_input_space = {pp_dict[name]['clip_min_input_space']} and "
                        f"clip_max_input_space = {pp_dict[name]['clip_max_input_space']} for field {name}."
                    )

            # Validation for preprocessed space clip ranges
            if (
                pp_dict[name]["clip_min_preproc_space"] is not None
                and pp_dict[name]["clip_max_preproc_space"] is not None
            ):
                if (
                    pp_dict[name]["clip_min_preproc_space"]
                    > pp_dict[name]["clip_max_preproc_space"]
                ):
                    raise ValueError(
                        "clip_min_preproc_space must be smaller than clip_max_preproc_space. "
                        f"You have clip_min_preproc_space = {pp_dict[name]['clip_min_preproc_space']} and "
                        f"clip_max_preproc_space = {pp_dict[name]['clip_max_preproc_space']} for field {name}."
                    )

            if pp_dict[name]["func"] is not None:
                if pp_dict[name]["inv_func"] is None:
                    raise ValueError(
                        "If a function is specified, an inverse function must also be specified."
                    )
            else:
                if pp_dict[name]["inv_func"] is not None:
                    raise ValueError(
                        "If an inverse function is specified, a function must also be specified."
                    )
        # apply selection cuts
        if pp_dict[name].get("larger_than") is not None:
            selection_mask = selection_mask & (ak_array[name] > pp_dict[name]["larger_than"])
        if pp_dict[name].get("smaller_than") is not None:
            selection_mask = selection_mask & (ak_array[name] < pp_dict[name]["smaller_than"])

    def _process_field_inverse(name, params):
        """Process a single field for inverse transformation."""
        # Step 1: Get the raw field value
        raw_value = getattr(ak_array, name)

        # Step 2: Undo scaling and shifting
        unscaled_value = raw_value / params["multiply_by"] + params["subtract_by"]

        # Step 3: Apply inverse function if present
        if params["inv_func"]:
            return eval(params["inv_func"])(unscaled_value)  # nosec
        else:
            return unscaled_value

    def _process_field_forward(name, params):
        """Process a single field for forward transformation."""
        # Step 1: Get the raw field value
        raw_value = getattr(ak_array, name)

        # Step 2: Apply input space clipping (if specified)
        if (
            params["clip_min_input_space"] is not None
            or params["clip_max_input_space"] is not None
        ):
            clipped_input_value = ak_clip(
                raw_value * 1.0, params["clip_min_input_space"], params["clip_max_input_space"]
            )
        else:
            clipped_input_value = raw_value

        # Step 3: Apply function transformation if present
        if params["func"] is not None:
            transformed_value = eval(params["func"])(clipped_input_value)  # nosec
        else:
            transformed_value = clipped_input_value

        # Step 4: Apply binning
        binned_value = apply_binning(transformed_value, params["bin_edges"])

        # Step 5: Apply selection mask
        masked_value = binned_value[selection_mask]

        # Step 6: Apply shifting and scaling
        shifted_value = masked_value - params["subtract_by"]
        scaled_value = shifted_value * params["multiply_by"]

        # Step 7: Apply preprocessed space clipping (if specified)
        if (
            params["clip_min_preproc_space"] is not None
            or params["clip_max_preproc_space"] is not None
        ):
            final_value = ak_clip(
                scaled_value, params["clip_min_preproc_space"], params["clip_max_preproc_space"]
            )
        else:
            final_value = scaled_value

        return final_value

    if inverse:
        return ak.Array(
            {name: _process_field_inverse(name, params) for name, params in pp_dict.items()}
        )

    return ak.Array(
        {name: _process_field_forward(name, params) for name, params in pp_dict.items()}
    )


# define a function to sort ak.Array by pt
def sort_by_pt(constituents: ak.Array, ascending: bool = False):
    """Sort ak.Array of jet constituents by the pt
    Args:
        constituents (ak.Array): constituents array that should be sorted by pt.
            It should have a pt attribute.
        ascending (bool, optional): If True, the first value in each sorted
            group will be smallest; if False, the order is from largest to
            smallest. Defaults to False.
    Returns:
        ak.Array: sorted constituents array
    """
    if isinstance(constituents, ak.Array):
        try:
            temppt = constituents.pt
        except AttributeError:
            raise AttributeError(
                "Trying to sort an ak.Array without a pt attribute. Please check the input."
            )
    indices = ak.argsort(temppt, axis=1, ascending=ascending)
    return constituents[indices]


def ak_smear(arr, sigma=0, seed=42):
    """Helper function to smear an array of values by a given sigma.

    Parameters
    ----------
    arr : awkward array
        The array to smear
    sigma : float, optional
        The sigma of the smearing, by default 0 (i.e. no smearing)
    seed : int, optional
        Seed for the random number generator, by default 42
    """
    # Convert it to a 1D numpy array and perform smearing
    numpy_arr = ak.to_numpy(arr.layout.content)

    if sigma != 0:
        rng = np.random.default_rng(seed)
        numpy_arr = rng.normal(numpy_arr, sigma)

    # Convert it back to awkward form
    return ak.Array(ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout))


def ak_clip(arr, clip_min=None, clip_max=None):
    """Helper function to clip the values of an array.

    Parameters
    ----------
    arr : awkward array
        The array to clip
    clip_min : float, optional
        Minimum value to clip to, by default None
    clip_max : float, optional
        Maximum value to clip to, by default None
    """
    ndim = arr.ndim
    # Convert it to a 1D numpy array and perform clipping
    if ndim > 1:
        numpy_arr = ak.to_numpy(arr.layout.content)
    else:
        numpy_arr = ak.to_numpy(arr)

    if clip_min is not None:
        numpy_arr = np.clip(numpy_arr, clip_min, None)

    if clip_max is not None:
        numpy_arr = np.clip(numpy_arr, None, clip_max)

    if ndim > 1:
        # Convert it back to awkward form
        return ak.Array(
            ak.contents.ListOffsetArray(arr.layout.offsets, ak.Array(numpy_arr).layout)
        )
    return numpy_arr


def ak_subtract(arr1, arr2):
    """Helper function to subtract two awkward arrays with names fields, i.e. arr1 - arr2

    Parameters
    ----------
    arr1 : ak.Array
        First array
    arr2 : ak.Array
        Second array

    Returns
    -------
    ak.Array
        Array with the fields of arr1 - arr2
    """

    if arr1.fields != arr2.fields:
        raise ValueError(
            "The two arrays do not have the same fields. Array 1 has fields: "
            f"{arr1.fields}, while array 2 has fields: {arr2.fields}"
        )

    if len(arr1) != len(arr2):
        raise ValueError(
            "The two arrays do not have the same length. Array 1 has length: "
            f"{len(arr1)}, while array 2 has length: {len(arr2)}"
        )

    if len(arr1.fields) == 0 or len(arr2.fields) == 0:
        raise ValueError("One or both arrays have no fields.")

    return ak.Array({name: getattr(arr1, name) - getattr(arr2, name) for name in arr1.fields})


def ak_mean(arr, axis=None):
    """Helper function to calculate the mean of an awkward array with field names along a certain
    axis.

    Parameters
    ----------
    arr : ak.Array
        Array to calculate the mean of
    axis : int, optional
        Axis along which to calculate the mean, by default None, which
        calculates the mean over all dimensions.

    Returns
    -------
    dict
        Dictionary with the mean of each field in the array. If the mean is still
        an awkward array, the values of the dict will be awkward arrays as well.
    """

    if not isinstance(arr, ak.Array):
        raise TypeError("Input arr must be an awkward array.")

    if axis is not None:
        if not isinstance(axis, int):
            raise TypeError("Input axis must be an integer.")
        # elif axis < 0 or axis >= arr.ndim:
        #     raise ValueError("Input axis is out of range.")

    return {name: ak.mean(getattr(arr, name), axis=axis) for name in arr.fields}


def ak_abs(arr):
    """Helper function to calculate the absolute value of an awkward array with field names.

    Parameters
    ----------
    arr : ak.Array
        Array to calculate the absolute value of

    Returns
    -------
    ak.Array
        Array with the absolute values of each field
    """

    if not isinstance(arr, ak.Array):
        raise TypeError("Input arr must be an awkward array.")
    if len(arr.fields) == 0:
        raise ValueError("Input arr has no fields.")

    return ak.Array({name: np.abs(getattr(arr, name)) for name in arr.fields})


def count_appearances(arr, mask, count_up_to: int = 10):
    """
    Parameters
    ----------
    arr : np.ndarray
        Array of integers, shape (n_jets, n_constituents)
    mask : np.ndarray
        Mask array, shape (n_jets, n_constituents)
    count_up_to : int, optional
        The maximum number of appearances to check for, by default 10

    Returns
    -------
    np.ndarray
        Array of shape (n_jets, n_tokens) containing the counts of each token.
        I.e. if the maximum token number is 5, the array will have 5 columns
        indicating how many times each token appears in each jet.
    np.ndarray
        Array of shape (n_jets, count_up_to) containing the number of tokens
        that appear 0, 1, 2, 3, ... times in each jet.
    np.ndarray
        Array of shape (n_jets, count_up_to) containing the fraction of tokens
        that appear 0, 1, 2, 3, ... times in each jet.
    """
    # fill the masked values with one above the maximum value in the array
    arr = np.where(mask != 0, arr, np.max(arr) + 1)

    # Count the occurrences of each integer in each row
    counts = np.array([np.bincount(row) for row in arr])
    # remove the last column, which is the count of the maximum (fill) value
    counts = counts[:, :-1]

    # calculate how many tokens appear 0, 1, 2, 3, ... times
    n_token_appearances = []
    for i in range(count_up_to + 1):
        n_token_appearances.append(np.sum(np.array(counts) == i, axis=1))

    # calculate the percentages of tokens that appear 0, 1, 2, 3, ... times
    n_tokens_total = np.sum(mask, axis=1)
    frac_token_appearances = np.array(
        [n * i / n_tokens_total for i, n in enumerate(n_token_appearances)]
    )

    return counts, np.array(n_token_appearances).T, frac_token_appearances.T


def calc_additional_kinematic_features(ak_particles, exclude_fields: list = None):
    """Takes in an awkward array of particles and calculates additional kinematic features. The
    initial ak array should contain part_pt, part_etarel, part_phirel.

    Parameters
    ----------
    ak_particles : ak.Array
        Array of particles with part_pt, part_etarel, part_phirel fields
    exclude_fields : list, optional
        List of fields to exclude from the output array, by default None

    Returns
    -------
    ak.Array
        Array with additional kinematic features
    """

    # check if all features are included
    required_fields = ["part_pt", "part_etarel", "part_phirel"]
    if not all([feat in ak_particles.fields for feat in required_fields]):
        raise ValueError(
            "Not all required features are present in the input array."
            f"Required features are: {required_fields}"
        )
    p4s_centered = ak.zip(
        {
            "pt": ak_particles.part_pt,
            "eta": ak_particles.part_etarel,
            "phi": ak_particles.part_phirel,
            "mass": ak_particles.part_mass
            if "part_mass" in ak_particles.fields
            else ak.zeros_like(ak_particles.part_pt),
        },
        with_name="Momentum4D",
    )
    p4s_jet = ak.sum(p4s_centered, axis=1)
    dict_for_ak_arr = {
        "part_px_centered": p4s_centered.px,
        "part_py_centered": p4s_centered.py,
        "part_pz_centered": p4s_centered.pz,
        "part_ptrel": ak_particles.part_pt / p4s_jet.pt,
        "part_energy_centered": p4s_centered.energy,
        "part_energy_centered_raw": p4s_centered.energy,  # this is a workaround to be able to use the energy twice (once for lorentz vectors, once as particle feature)
        "part_erel_centered": p4s_centered.energy / p4s_jet.energy,
        "part_deltaR": p4s_centered.deltaR(p4s_jet),
    }
    logger.info(f"Dict keys for additional kinematic features: {list(dict_for_ak_arr.keys())}")
    if exclude_fields is not None:
        logger.info(f"Excluding fields: {exclude_fields}")
        for field in exclude_fields:
            if field in dict_for_ak_arr:
                dict_for_ak_arr.pop(field)
        logger.info(f"Remaining fields: {list(dict_for_ak_arr.keys())}")
    return ak.Array(dict_for_ak_arr)


def apply_binning(arr, bin_edges, return_bin_centers=False):
    """Helper function to apply a certain binning to an array. Values outside the bin edges are
    clipped to the bin edges.

    Parameters
    ----------
    arr : np.ndarray or ak.Array
        Array to bin. If ak.Array, the array is flattened and then unflattened again.
    bin_edges : np.ndarray
        Array of bin edges.
    return_bin_centers : bool, optional
        If True, also return the bin centers. Default is False.

    Returns
    -------
    np.ndarray
        Binned array. If bin_edges is None, the input array is returned.
    np.ndarray
        Bin centers. Only returned if return_bin_centers is True.
    """
    if bin_edges is None:
        return arr
    # flatten the array to use numpy functions
    # check if it is an awkward array with nested structure
    counts = None
    if isinstance(arr, ak.Array):
        if arr.ndim > 1:
            counts = ak.num(arr)
            arr = ak.flatten(arr)
    # clip the values to the bin edges
    arr = np.clip(arr, bin_edges[0], bin_edges[-1])
    bin_indices = np.digitize(arr, bin_edges, right=False)
    # make max index one smaller than the number of bins
    bin_indices = np.where(bin_indices == len(bin_edges), len(bin_edges) - 1, bin_indices)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Replace each bin index with the corresponding bin center
    binned_arr = bin_centers[bin_indices - 1]
    if counts is not None:
        binned_arr = ak.unflatten(binned_arr, counts)
    if return_bin_centers:
        return binned_arr, bin_centers
    return binned_arr


def convert_torch_token_sequence_with_stop_token_to_ak(tensor, stop_value):
    """Convert a torch tensor with sequences to an awkward array of variable length sequences using
    the stop_value. The stop_value indicates the end of the sequence.

    E.g. if the input tensor is:
    ```
    [
        [0, 2, 1, 7, 9],
        [0, 4, 7, 9, 0],
    ]
    ```
    Then the output will be (assuming `stop_value=7`):
    ```
    [
        [0, 2, 1],
        [0, 4],
    ]
    ```

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with sequences.
    stop_value : float
        Value that indicates the end of the sequence.

    Returns
    -------
    ak.Array
        Array of variable length sequences.
    """

    tensor_np = tensor.unsqueeze(-1).detach().cpu().numpy()
    mask = np.zeros((tensor.size(0), tensor.size(1)))
    # find where the stop value is in the sequences
    stop_indices = (tensor_np[:, :, 0] == stop_value).argmax(axis=1)
    # Correct indices where the stop value might not be present (argmax returns 0 in such cases)
    no_stop_value = ~(tensor_np[:, :, 0] == stop_value).any(axis=1)
    stop_indices[no_stop_value] = tensor_np.shape[
        1
    ]  # Use the sequence length for sequences without the stop value

    # build the mask
    for idx, sequence in enumerate(tensor_np):
        mask[idx, : stop_indices[idx]] = 1

    # convert to awkward array
    ak_arr = np_to_ak(tensor_np, names=["dummy"], mask=mask)

    return ak_arr["dummy"]


def arctanh_with_delta(x, delta=1e-5):
    """Arctanh with a small delta to avoid dealing with infinities. This is useful for using
    arctanh as a preprocessing function.

    Parameters
    ----------
    x : ak.Array or np.ndarray
        Input value.
    delta : float, optional
        Small delta to avoid infinities. Default is 1e-5.
    """
    return np.arctanh(ak_clip(x, clip_min=-1 + delta, clip_max=1 - delta))


def fix_padded_logits(logits, mask, factor=1e6):
    """Used to fix a tensor of logits if the sequences are padded after some token.
    The logits of the padded values are all set to 0, except for the first
    value, which is set to `factor`. This is useful when using the logits to
    calculate the loss.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of logits. Shape (batch_size, seq_len, n_tokens)
    mask : torch.Tensor
        Mask tensor. Shape (batch_size, seq_len). Must of type bool.
    factor : float, optional
        Value to set the first token of the padded values to. Default is 1e6.

    Returns
    -------
    torch.Tensor
        Fixed logits.
    """
    # fix the padded logits
    logits = logits * mask.unsqueeze(dim=-1)
    # set the logits of padded values to [1e6, -1e6, -1e6, ...]
    logits = logits + torch.cat(
        [
            (~mask).unsqueeze(-1) * factor,
            torch.zeros_like(logits[:, :, 1:]),
        ],
        dim=-1,
    )
    return logits


def ak_add_zero_padded_features(ak_arr, add_start=False, add_end=False):
    """
    Adds zero-padded features along the first axis, either at the beginning, end, or both.

    Parameters
    ----------
    ak_arr : ak.Array
        Awkward array to modify
    add_start : bool, optional
        Whether to add zero-padded features at the beginning, by default False
    add_end : bool, optional
        Whether to add zero-padded features at the end, by default False

    Returns
    -------
    ak.Array
    """

    # TODO: add a unit test for this function

    if add_start and add_end:
        return ak.concatenate(
            [ak.zeros_like(ak_arr[:, :1]), ak_arr, ak.zeros_like(ak_arr[:, :1])], axis=1
        )
    elif add_start:
        return ak.concatenate([ak.zeros_like(ak_arr[:, :1]), ak_arr], axis=1)
    elif add_end:
        return ak.concatenate([ak_arr, ak.zeros_like(ak_arr[:, :1])], axis=1)
    else:
        raise ValueError("Either `add_start` or `add_end` (or both) has to be set to true.")


def shuffle_ak_arr_along_axis1(ak_arr, seed=None):
    """Shuffle an awkward array along the first axis.

    Parameters
    ----------
    ak_arr : ak.Array
        Array to shuffle.
    seed : int, optional
        Seed for the random number generator, by default None
    """
    names = ak_arr.fields
    maxlen = int(ak.max(ak.num(ak_arr[names[0]])))
    ak_arr_padded, mask = ak_pad(x=ak_arr, maxlen=maxlen, axis=1, fill_value=0, return_mask=True)
    mask = mask.to_numpy()
    np_arr = ak_to_np_stack(ak_arr_padded, names=names)

    # shuffle each row of the array
    # this implementation is not the most efficient, but to me it seems that
    # rng.shuffle, rng.permutation and rng.permuted all apply the same permutation
    # (or don't apply the same permutation to all features simultaneously)
    rng = np.random.default_rng(seed)
    shuffled_arr = np.empty_like(np_arr)
    shuffled_mask = np.empty_like(mask)
    for i in range(np_arr.shape[0]):
        indices = rng.permutation(np_arr.shape[1])
        shuffled_arr[i] = np_arr[i, indices]
        shuffled_mask[i] = mask[i, indices]

    ak_arr_permuted = np_to_ak(shuffled_arr, names=names, mask=shuffled_mask)
    return ak_arr_permuted


def set_fraction_ones_to_zeros(mask: torch.Tensor, fraction: torch.Tensor):
    """Set a fraction of the ones in a mask to zero. A new mask is returned while
    the input is kept unchanged.

    Parameters
    ----------
    mask : torch.Tensor
        Mask tensor.
    fraction : float
        Fraction of ones to set to zero

    Returns
    -------
    torch.Tensor
        Modified mask tensor
    """
    mask = mask.clone()
    ones_indices = (mask == 1).nonzero(as_tuple=True)
    num_ones = len(ones_indices[0])
    num_to_zero = int(num_ones * fraction)
    indices_to_zero = torch.randperm(num_ones)[:num_to_zero]
    mask[ones_indices[0][indices_to_zero], ones_indices[1][indices_to_zero]] = 0
    return mask


def concat_up_to_n(arr_list: list[np.ndarray], n: int):
    """
    Concatenates arrays from a list up to a total of `n` elements along the first axis.
    Parameters
    ----------
    arr_list : list of np.ndarray
        List of NumPy arrays to concatenate. Each array must have at least one dimension.
    n : int
        The maximum number of elements to concatenate along the first axis.
    Returns
    -------
    np.ndarray or None
        A concatenated NumPy array containing up to `n` elements along the first axis.
        Returns None if the input list is empty or if no arrays are concatenated.
    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5])
    >>> concat_up_to_n([a, b], 4)
    array([1, 2, 3, 4])
    """

    out = []
    count = 0
    for arr in arr_list:
        if count >= n:
            break
        remaining = n - count
        if arr.shape[0] <= remaining:
            out.append(arr)
            count += arr.shape[0]
        else:
            out.append(arr[:remaining])
            count += remaining
    if out:
        return np.concatenate(out)
    else:
        return None


def replace_masked_positions(
    x: torch.Tensor,
    mask_is_valid: torch.Tensor,
    mask_is_valid_corrupted: torch.Tensor,
    mask_is_valid_but_masked: torch.Tensor,
    vectors_to_insert: torch.Tensor,
    pos_encoding_type: str,
    pos_encoding_feature: torch.Tensor = None,
):
    supported_pos_enc_types = [
        None,
        "sort_descending_all",
        "sort_descending_in_masked_subset",
    ]
    if pos_encoding_type not in supported_pos_enc_types:
        raise ValueError(
            "Positional encoding type not supported. Supported types are: "
            f"{supported_pos_enc_types}, but got {pos_encoding_type}"
        )

    # check that the feature dimension (last) matches
    if x.shape[-1] != vectors_to_insert.shape[-1]:
        raise ValueError(
            "Feature dimension (last) of x and vectors_to_insert must match. "
            f"Got {x.shape[-1]} and {vectors_to_insert.shape[-1]}"
        )
    # check that masks have correct shape (should have the same len in dim 0 and 1)
    if x.shape[0] != mask_is_valid.shape[0] or x.shape[1] != mask_is_valid.shape[1]:
        raise ValueError(
            f"x and mask_is_valid have incompatible shape:{x.shape} and {mask_is_valid.shape}"
        )

    if pos_encoding_type is not None and pos_encoding_feature is None:
        raise ValueError(
            "Positional encoding feature must be provided if pos_encoding_type is not None."
        )

    if pos_encoding_feature is not None:
        if mask_is_valid.shape != pos_encoding_feature.shape:
            raise ValueError(
                "Shape of mask_is_valid and pos_encoding_feature must match. "
                f"Got {mask_is_valid.shape} and {pos_encoding_feature.shape}"
            )

    # check that there are enough vectors to insert
    max_n_points = mask_is_valid.shape[1]
    n_vectors_to_insert = vectors_to_insert.shape[0]
    if n_vectors_to_insert < max_n_points:
        raise ValueError(
            "Not enough vectors to insert. "
            f"Got {n_vectors_to_insert}, but need at least {max_n_points}."
        )

    indices = torch.nonzero(mask_is_valid_but_masked, as_tuple=True)

    if pos_encoding_type is None:
        x[indices] = vectors_to_insert[indices[1]]
    elif pos_encoding_type in [
        "sort_descending_all",
        "sort_descending_in_masked_subset",
    ]:
        # ensure the sort feature is available and clone it
        pos_encoding_feature = pos_encoding_feature.clone()
        # set sorting feature of invalid particles to -inf to ensure they are at
        # the end after sorting (and thus don't affect sorting)
        pos_encoding_feature[mask_is_valid == 0] = float("-inf")

        if pos_encoding_type == "sort_descending_in_masked_subset":
            # set sorting feature of valid but unmasked particles to -inf to ensure they are at
            # the end after sorting (and thus don't affect sorting)
            pos_encoding_feature[mask_is_valid_corrupted == 1] = float("-inf")

        sort_index = torch.argsort(pos_encoding_feature, dim=1, descending=True)
        rank_index = torch.argsort(sort_index, dim=1)

        # replace masked entries in x with vectors_to_insert
        x[indices] = vectors_to_insert[rank_index[indices]]
    else:
        raise ValueError("This should never happen.")

    return None
