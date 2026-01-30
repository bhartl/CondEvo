import os
import h5py
import numpy as np
import pandas as pd

# local import
import utils


def load_benchmark(es="HADES", objective="rastrigin", run_ids=None, dst=utils.DST):
    """ load run data as pandas dataframe """
    h5_filename = os.path.join(dst, utils.H5_FILE.format(ES=es, objective=objective))

    h5_files = []
    h5_basename = os.path.basename(h5_filename.replace(".h5", ""))
    for filename in os.listdir(os.path.dirname(h5_filename)):
        if h5_basename in filename:
            h5_files.append(os.path.join(os.path.dirname(h5_filename), filename))

    h5_data = []
    run_offset = 0
    for h5_filename in h5_files:
        print("loading data from", h5_filename)
        with h5py.File(h5_filename, 'r') as f:
            if run_ids is None:
                run_ids = list(sorted([int(k.replace("run_", "")) for k in f.keys()]))

            if not isinstance(run_ids, list):
                run_ids = [run_ids]

            try:
                data = []
                print("found runs:", run_ids)
                for run_id in run_ids:
                    run_data = f[f"run_{run_id}"]
                    gen_ids = list(sorted([int(k.replace("gen_", "")) for k in run_data.keys()]))

                    # extract attributes
                    run_attrs = run_data.attrs
                    run_dict = {}
                    for k, v in run_attrs.items():
                        try:
                            v = json.loads(v)
                        except:
                            pass
                        run_dict[k] = v

                    for gen_id in gen_ids:
                        generation = run_data[f"gen_{gen_id}"]

                        try:
                            samples = generation['samples'][()]
                            fitness = generation['fitness'][()]

                            item = {
                                "run": run_id + run_offset,
                                "gen": gen_id,
                                "samples": samples,
                                "fitness": fitness,
                                **run_dict
                            }
                            if 'model_loss' in generation:
                                model_loss = generation['model_loss'][()]
                                item['model_loss'] = model_loss

                            # add generation data and attribute data as data-element for dataframe
                            data.append(item)

                        except KeyError:
                            pass

                run_offset += len(run_ids)
                h5_data.extend(data)

            except KeyError:
                print("no runs found in file", h5_filename)

    return pd.DataFrame(h5_data)


def plot_fitness(df, run_id=None, figsize=(12, 6), ax=None):
    import matplotlib.pyplot as plt

    if run_id is not None:
        df = df[df.run == run_id]

    for run, df_run in df.groupby("run"):
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        ax.plot(df_run.gen, np.max([v for v in df_run.fitness], axis=-1), label=f"Max. Fitness (Run {run})")
        ax.plot(df_run.gen, np.mean([v for v in df_run.fitness], axis=-1), label=f"Avg. Fitness (Run {run})", linestyle="--")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid()

    return ax


if __name__ == '__main__':
    import argh
    argh.dispatch_commands([load_benchmark,
                            plot_fitness])
