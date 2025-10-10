import concurrent
import functools
import traceback  # To get detailed error messages
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from time import time
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

# isort: off
from src.mma.constants import (
    CHUNK_SIZE_PARALLEL_PROCESSING,
    DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
)

_logger = structlog.getLogger()

R = TypeVar("R")
_GDF_INDEX_COLUMN_NAME = "_parallel_gdf_index"


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
    split_gdf: List[pd.DataFrame],
    max_workers: int = DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    reset_index: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generic wrapper function to parallelize the processing of a list with split GeoDataFrames with
    the given function and its arguments using ProcessPoolExecutor

    Args:
        :param input_function: The function to be executed in parallel. This function should take
        arguments as positional arguments
        :param split_gdf: The list with split GeoDataFrames
        :param reset_index: If "true", the index of the merged GeoDataFrames is reset
        :param kwargs: The Keyword Arguments
        :param max_workers: Maximum number of parallel processes to use in the ProcessPoolExecutor

    :return:
        The processed and merged GeoDataFrame
    """

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # start the input function with the given GeoDataFrame and keyword arguments and mark each
        # future object with its processed GeoDataFrame

        result = {executor.submit(input_function, gdf, **kwargs): gdf for gdf in split_gdf}
        gdf_list = []
        for future in tqdm(concurrent.futures.as_completed(result), total=len(split_gdf)):
            # for future in result:
            # gdf_list.append(future.result())
            try:
                gdf_list.append(future.result())
            except Exception as e:
                print(f"Error processing GeoDataFrame at index: {e}")
                traceback.print_exc()  # Print full error details for debugging

    output_gdf = pd.concat(gdf_list)
    if reset_index:
        output_gdf.reset_index(drop=True, inplace=True)

    return output_gdf


def parallelize_dataframe(
    input_function: Callable[..., R],
    gdf: pd.DataFrame,
    max_workers: int = DEFAULT_MAX_WORKERS_PARALLEL_PROCESSING,
    chunk_size: int = CHUNK_SIZE_PARALLEL_PROCESSING,
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
        :param gdf: The GeoDataFrame
        :param kwargs: The Keyword Arguments
        :param max_workers: Maximum number of parallel processes to use in the ProcessPoolExecutor
        :param chunk_size: The chunk size for parallel processing

    :return:
        The processed and merged GeoDataFrame
    """

    if gdf.empty:
        return gdf

    number_of_chunks = ceil(len(gdf) / chunk_size)
    if max_workers > 1 and number_of_chunks > 2:
        split_gdf_list = df_split(
            df=gdf,
            chunks=int(len(gdf) / chunk_size),
            index_column=_GDF_INDEX_COLUMN_NAME,
        )
        output_gdf = gdf_list_process_pool_executor(
            input_function, split_gdf=split_gdf_list, max_workers=max_workers, **kwargs
        )
        if len(output_gdf) > 0:
            output_gdf = output_gdf.reset_index(drop=True)
            if _GDF_INDEX_COLUMN_NAME in output_gdf.columns:
                output_gdf = output_gdf.set_index(output_gdf[_GDF_INDEX_COLUMN_NAME]).sort_index()
                output_gdf = output_gdf.drop(
                    columns=[_GDF_INDEX_COLUMN_NAME], errors="ignore"
                ).copy()
            return output_gdf
    else:
        # execute func directly without splitting the Dataframe and using the process pool executor
        output_gdf = input_function(gdf, **kwargs)
        return output_gdf

    output_gdf = gdf[0:0]
    return output_gdf


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
    df_list: List[pd.DataFrame] = [df for df in np.array_split(df, chunks) if len(df) > 0]
    return df_list
