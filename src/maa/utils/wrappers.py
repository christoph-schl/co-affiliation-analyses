# Copyright © 2025 Christoph Schlager, TU Wien

import concurrent
import functools
import traceback  # To get detailed error messages
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from time import time
from typing import Any, Callable, List, Optional, TypeVar, Union

import geopandas as gpd
import pandas as pd
import structlog
from tqdm import tqdm

# isort: off
from maa.constants.constants import (
    CHUNK_SIZE_PARALLEL_PROCESSING,
    DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
)

_logger = structlog.getLogger(__name__)

R = TypeVar("R")
_DF_INDEX_COLUMN_NAME = "_parallel_gdf_index"


def get_execution_time(input_function: Callable[..., R]) -> Callable[..., R]:
    """
    Wrapper function to estimate the execution time of the given function
    :param input_function: The function for which the execution time should be measured
    """

    @functools.wraps(input_function)
    def get_execution_time_wrapper(*args: Any, **kwargs: Any) -> Any:
        t1 = time()
        _logger.info(f"{input_function.__name__!r} started")
        result = input_function(*args, **kwargs)
        t2 = time()
        execution_time = round(t2 - t1, 4)
        _logger.info(f"{input_function.__name__!r} executed", runtime_s=execution_time)
        return result

    return get_execution_time_wrapper


def gdf_list_process_pool_executor(
    input_function: Callable[..., R],
    split_df: List[Union[pd.DataFrame, gpd.GeoDataFrame]],
    max_workers: int = DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    reset_index: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Generic wrapper function to parallelize the processing of a list with split GeoDataFrames with
    the given function and its arguments using ProcessPoolExecutor

    Args:
        :param input_function: The function to be executed in parallel. This function should take
        arguments as positional arguments
        :param split_df: The list with split GeoDataFrames
        :param reset_index: If "true", the index of the merged GeoDataFrames is reset
        :param kwargs: The Keyword Arguments
        :param max_workers: Maximum number of parallel processes to use in the ProcessPoolExecutor
        :param verbose: Shows a progress bar, if true

    :return:
        The processed and merged GeoDataFrame
    """

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # start the input function with the given GeoDataFrame and keyword arguments and mark each
        # future object with its processed GeoDataFrame

        result = {executor.submit(input_function, df, **kwargs): df for df in split_df}
        df_list = []
        for future in tqdm(
            concurrent.futures.as_completed(result), total=len(split_df), disable=not verbose
        ):
            try:
                df_list.append(future.result())
            except Exception as e:
                print(f"Error processing GeoDataFrame at index: {e}")
                traceback.print_exc()  # Print full error details for debugging

    output_df = pd.concat(df_list)
    if reset_index:
        output_df.reset_index(drop=True, inplace=True)

    return output_df


def parallelize_dataframe(
    input_function: Callable[..., R],
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    max_workers: int = DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    chunk_size: int = CHUNK_SIZE_PARALLEL_PROCESSING,
    verbose: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generic wrapper function to parallelize the processing of a GeoDataFrame with the given function
    and its arguments. It splits the geopandas GeoDataFrame into chunks and passes the list of split
    GeoDataFrame to the gdf_list_process_pool_executor function which handles the processing of a
    list of GeoDataFrames based on the concurrent futures ProcessPoolExecutor.
    This wrapper can be used for simple functions where no dependencies between columns in the
    GeoDataFrame need to be
    considered


    Args:
        :param input_function: The function to be executed in parallel. This function should take
        arguments as positional arguments
        :param df: The DataFrame or GeoDataFrame
        :param kwargs: The Keyword Arguments
        :param max_workers: Maximum number of parallel processes to use in the ProcessPoolExecutor
        :param chunk_size: The chunk size for parallel processing
        :param verbose: Shows a process bar, if true

    :return:
        The processed and merged GeoDataFrame
    """

    if df.empty:
        return df

    number_of_chunks = ceil(len(df) / chunk_size)
    if max_workers > 1 and number_of_chunks > 2:
        split_gdf_list = df_split(
            df=df,
            chunks=int(len(df) / chunk_size),
            index_column=_DF_INDEX_COLUMN_NAME,
        )
        output_df = gdf_list_process_pool_executor(
            input_function,
            split_df=split_gdf_list,
            max_workers=max_workers,
            verbose=verbose,
            **kwargs,
        )
        if len(output_df) > 0:
            output_df = output_df.reset_index(drop=True)
            if _DF_INDEX_COLUMN_NAME in output_df.columns:
                output_df = output_df.set_index(output_df[_DF_INDEX_COLUMN_NAME]).sort_index()
                output_df = output_df.drop(columns=[_DF_INDEX_COLUMN_NAME], errors="ignore").copy()
            return output_df
    else:
        # execute func directly without splitting the Dataframe and using the process pool executor
        output_df = input_function(df, **kwargs)
        return output_df

    output_df = df[0:0]
    return output_df


def _split_dataframe(
    df: Union[pd.DataFrame, gpd.GeoDataFrame], chunks: int
) -> List[Union[pd.DataFrame, gpd.GeoDataFrame]]:
    n = len(df)
    chunk_size = n // chunks + (n % chunks > 0)
    df_list = [df.iloc[i : i + chunk_size] for i in range(0, n, chunk_size)]  # noqa: E203
    return df_list


def df_split(
    df: pd.DataFrame, chunks: int, index_column: Optional[str] = None
) -> List[pd.DataFrame]:
    """
    Splits the GeoDataFrame into a list of GeoDataFrames for a given chunk size
        :param df: The GeoDataFrame
        :param chunks: The number of chunks
        :param index_column: The column name with index values of the GeoDataFrame
        :return: The list of GeoDataFrames

    """
    df = df.copy()
    if chunks == 0:
        chunks = 1
    if index_column is not None:
        df[index_column] = df.index.values
    # df_list: List[pd.DataFrame] = [df for df in np.array_split(df, chunks) if len(df) > 0]

    df_list = [chunk for chunk in _split_dataframe(df, chunks) if len(chunk) > 0]

    return df_list
