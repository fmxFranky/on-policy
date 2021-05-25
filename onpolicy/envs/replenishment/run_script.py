'''
parser.add_argument("--run-name", type=str, default='0110_dqn_pretrain_OC0_H10_3')

parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--use-unc-part", default=True, type=bool_map)
parser.add_argument("--pretrain", default=False, type=bool_map)
parser.add_argument("--fixed-uncontrollable-param", default=False, type=bool_map)
parser.add_argument("--demand-random-noise", default=False, type=bool_map)
parser.add_argument("--use-cnn-state", default=False, type=bool_map)
parser.add_argument("--pretrain-epoch", default=100, type=int)
parser.add_argument("--embeddingmerge", default='cat', type=str, help="'cat' or 'dot'")
parser.add_argument("--activation-func", default='sigmoid', type=str, help="'sigmoid', 'relu' or 'tanh'")
parser.add_argument("--use-bn", default=False, type=bool_map)
parser.add_argument("--num-iterations", default=100, type=int)
parser.add_argument("--seed", default=1, type=int)
'''
import os
import random


def base_setting(pretrain=False):
  bash = f" --pretrain {pretrain}"
  bash += f" --fixed-uncontrollable-param {pretrain}"
  if pretrain:
    name = "_pretrain"
  else:
    name = "_baseline"
  return bash, name


def pretrain_epoch(epoch=100):
  bash = f" --pretrain-epoch {epoch}"
  name = f"_pret{epoch}"
  return bash, name


def weight_decay_setting(decay=0.2):
  bash = f" --weight-decay {decay}"
  name = f"_decay{decay}"
  return bash, name


def lr_setting(lr=0.0003):
  bash = f" --lr {lr}"
  name = f"_lr{lr}"
  return bash, name


def actfunc(act='sigmoid'):
  bash = f" --activation-func {act}"
  name = f"_{act}"
  return bash, name


def oracle_setting():
  bash = f" --oracle True"
  name = f"_oracle"
  return bash, name


def train_augmentation_setting(augmentation='none'):
  bash = f" --train-augmentation {augmentation}"
  name = f"_Aug{augmentation}"
  return bash, name


def setting_add(bash, name, bn):
  b, n = bn
  return bash + b, name + n


def to_bash(bash, name):
  if not os.path.exists('output_log'):
    os.mkdir('output_log')

  return f"nohup {bash} --run-name {name} > output_log/{name}.txt  2>&1 &\n"


def sleep_xs(x):
  return f"sleep {x}s\n"


# nohup python -u test.py > out.log 2>&1 &
# -u 实时刷新缓冲区


def run_x_seed(
    x,
    f,
    pretrain=False,
    act='sigmoid',
    # pre_epoch=300,
    augmentation='none',
    oracle=False,
    weight_decay=0.0,
    lr=0.0003):
  bash = "python -u inventory_train_dqn_only_store.py"
  name = "2021-01-23_H5_OC100_storage1000"
  bash, name = setting_add(bash, name, base_setting(pretrain))
  if oracle:
    bash, name = setting_add(bash, name, oracle_setting())
  if act != 'sigmoid':
    bash, name = setting_add(bash, name, actfunc(act))
  if weight_decay != 0.0:
    bash, name = setting_add(bash, name, weight_decay_setting(weight_decay))
  if lr != 0.0003:
    bash, name = setting_add(bash, name, lr_setting(lr))

  # if pretrain:
  #     bash, name = setting_add(bash, name, pretrain_epoch(pre_epoch))
  bash, name = setting_add(bash, name, train_augmentation_setting(augmentation))

  for i in range(x):
    seed = random.randint(1, 10000)
    n_bash, n_name = setting_add(bash, name,
                                 (f" --seed {seed}", f"_seed{seed}"))
    f.write(to_bash(n_bash, n_name))

  f.write(sleep_xs(2))


if __name__ == '__main__':
  f = open("run.sh", 'w')

  run_x_seed(4, f, pretrain=False, augmentation='none', weight_decay=0.0)
  run_x_seed(4, f, pretrain=True, augmentation='dense', weight_decay=0.0)
  run_x_seed(4, f, pretrain=True, augmentation='sparse', weight_decay=0.0)
  run_x_seed(4, f, pretrain=False, oracle=True)

  f.close()
