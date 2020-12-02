import os
import subprocess

import ray

if __name__=='__main__':
    package_home = os.path.abspath(os.path.join(ray.__file__, '../rllib/policy/rnn_sequencing.py'))
    rnn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rnn_sequencing.py'))
    print('We are replacing the rnn_sequence file in {} with {}'.format(package_home, rnn_path))
    subprocess.check_call(["rm", "-rf", package_home])
    subprocess.check_call(["cp", rnn_path, package_home])
