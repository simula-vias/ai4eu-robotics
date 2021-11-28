import click
import scipy.signal
import scipy.fft
import numpy as np
import pandas as pd
import os

import csv
import gzip


def chunk_generator(data_file):
    with gzip.open(data_file, "rt") as f:
        csvr = csv.DictReader(f)

        chunk = []
        chunk_key = None

        is_wrist = False

        for r in csvr:
            is_wrist = "s2" in r

            if is_wrist:
                row_key = r["key"]
            else:
                row_key = (r["key"], r["phase"])

            if chunk_key is None:
                chunk_key = row_key
                chunk = [r]
            elif chunk_key == row_key:
                chunk.append(r)
            else:
                # Chunk completed
                df = pd.DataFrame().from_records(chunk)
                df = df[~df["s1"].isna()]
                if is_wrist:
                    df = df[~df["s2"].isna()]
                yield df

                # Start new chunk
                chunk_key = row_key
                chunk = [r]

        df = pd.DataFrame().from_records(chunk)
        df = df[~df["s1"].isna()]
        if is_wrist:
            df = df[~df["s2"].isna()]
        yield df


def fft(values):
    data = values - np.mean(values)
    yf = scipy.fft.rfft(data).real
    # xf = scipy.fft.rfftfreq(data.size, 1 / len(chunk)).round().astype(int)
    return np.abs(yf).round(5)


@click.command()
@click.argument("input_files", nargs=-1)
@click.argument("output_directory", nargs=1)
@click.option(
    "-s", "--samples", default=6144, help="Samples per second; affects output shape"
)
@click.option(
    "-f",
    "--format",
    default="raw",
    type=click.Choice(["raw", "fft"], case_sensitive=False),
)
def convert(input_files, output_directory, samples, format):
    print(input_files, output_directory, samples, format)

    for data_file in input_files:
        convert_single_file(data_file, output_directory, samples, format)


def convert_single_file(input_file, output_directory, samples, format):
    filename = os.path.basename(input_file).replace(".csv", "").replace(".gz", "")
    print(filename)
    out_file = os.path.join(output_directory, f"{filename}_{format}.csv")

    with open(out_file, "w") as outf:
        csvout = None

        for chunk in chunk_generator(input_file):
            if len(chunk) == 0:
                continue

            is_wrist = "s2" in chunk.columns
            sensor_values = {}

            for sensor in ("s1", "s2"):
                if not is_wrist:
                    continue

                if len(chunk) != samples:
                    sensor_values[sensor] = scipy.signal.resample(
                        chunk[sensor].values, samples
                    ).reshape((samples,))
                else:
                    sensor_values[sensor] = chunk[sensor].values

                if format == "fft":
                    sensor_values[sensor] = fft(sensor_values[sensor].astype('float'))


            if is_wrist:
                new_row = {
                    "index": chunk["key"][0],
                    "scenario": chunk["scenario"][0],
                    "movement": chunk["movement"][0],
                    "iteration": chunk["iteration"][0],
                }
            else:
                chunk_key = chunk["key"][0]
                chunk_phase = chunk["phase"][0]
                new_row = {
                    "index": f"{chunk_key}_{chunk_phase}",
                    "phase": chunk_phase,
                    "pattern": chunk["pattern"][0],
                    "iteration": chunk["iteration"][0],
                }

            for k, v in sensor_values.items():
                new_row.update(**{f"{k}_{i}": v for i, v in enumerate(v)})

            if csvout is None:
                csvout = csv.DictWriter(outf, fieldnames=new_row.keys())
                csvout.writeheader()

            csvout.writerow(new_row)


if __name__ == "__main__":
    convert()
