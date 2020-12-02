import os
import ray
from flow.visualize.bottleneck_results import run_bottleneck_results
import subprocess
import errno


def aws_sync(src, dest):
    print('AWS S3 SYNC FROM >>> {} <<< TO >>> {} <<<'.format(src, dest))
    for _ in range(4):
        try:
            p1 = subprocess.Popen("aws s3 sync {} {}".format(src, dest).split(' '))
            p1.wait(60)
        except Exception as e:
            print('This is the error ', e)


EXP_TITLE_LIST = ["i2400_td3_senv_0p1_h400_reroute_rwd_e3_check",
                  "i2400_td3_senv_0p2_h400_reroute_rwd_e3",
                  "i2400_td3_senv_0p4_h400_reroute_rwd_e3"]
NUM_TEST_TRIALS = 20

OUTFLOW_MIN = 400
OUTFLOW_MAX = 3600
OUTFLOW_STEP = 100

DATE = "07-14-2020"


if __name__ == '__main__':
    # download checkpoints from AWS
    os.makedirs(os.path.expanduser("~/ray_results"))
    aws_sync('s3://nathan.experiments/trb_bottleneck_paper/07-13-2020',
             os.path.expanduser("~/ray_results"))

    ray.init()

    for EXP_TITLE in EXP_TITLE_LIST:
        # create output dir
        output_path = os.path.join(os.path.expanduser('~/bottleneck_results'), EXP_TITLE)
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        # for each grid search, find checkpoint 350
        for (dirpath, dirnames, filenames) in os.walk(os.path.expanduser("~/ray_results")):
            if "checkpoint_350" in dirpath and dirpath.split('/')[-3] == EXP_TITLE:
                print('FOUND CHECKPOINT {}'.format(dirpath))

                # grab the experiment name
                folder = os.path.dirname(dirpath)
                tune_name = folder.split("/")[-1]
                checkpoint_path = os.path.dirname(dirpath)

                print('GENERATING GRAPHS')
                run_bottleneck_results(OUTFLOW_MIN, OUTFLOW_MAX, OUTFLOW_STEP, NUM_TEST_TRIALS, output_path, EXP_TITLE, checkpoint_path,
                                        gen_emission=False, render_mode='no_render', checkpoint_num="350",
                                        horizon=400, end_len=500)

                aws_sync(output_path,
                        "s3://nathan.experiments/trb_bottleneck_paper/graphs_test/{}/{}/{}".format(DATE, EXP_TITLE, tune_name))
