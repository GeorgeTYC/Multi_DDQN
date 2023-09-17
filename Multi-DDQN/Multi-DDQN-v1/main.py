import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.backend import set_session
from utils.Enviroment import Enviroment

from utils.DDQN import DDQN

def parse_args(args):
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--type', type=str, default='DDQN', help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")

    # TODO：修改默认值
    parser.add_argument('--nb_episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--with_PER', dest='with_per', action='store_true',
                        help="Use Prioritized Experience Replay (DDQN + PER)")

    parser.add_argument('--gather_stats', default=True ,dest='gather_stats', action='store_true',
                        help="Compute Average reward per episode (slower)")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    return parser.parse_args(args)

def main(args=None):
    args = parse_args(args)

    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True

    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(tf.compat.v1.Session(config=config))
    summary_writer = tf.summary.create_file_writer(args.type + "/tensorboard_" + "Writen_By_Myself_v3")

    env = Enviroment()
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    algo = DDQN(state_dim=state_dim, action_dim=action_dim, args=args)

    stats = algo.train(env, args, summary_writer)

    # Export results to CSV
    if (args.gather_stats):
        df = pd.DataFrame(np.array(stats))
        df.to_csv(args.type + "/logs_v3.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(args.type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_NB_EP_{}_BS_{}'.format(exp_dir,
                                            args.type,
                                            args.nb_episodes,
                                            args.batch_size)

    algo.save_weights(export_path)

if __name__=="__main__":
    main()