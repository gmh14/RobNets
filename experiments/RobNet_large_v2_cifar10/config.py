# Model Params
model = 'robnet_large_v2'
model_param = dict(C=64,
                   num_classes=10,
                   layers=33,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=True,
                   AdPoolSize=1)

# Dataset Params
dataset = 'cifar10'
dataset_param = dict(data_root='../data/cifar10',
                     batch_size=32, # With a world_size of 32, the total batch_size is 1024 for training
                     num_workers=2)
report_freq = 10
seed = 10
save_path = './log'
resume_path = dict(path='./checkpoint/RobNet_large_v2_cifar10.pth.tar', origin_ckpt=True)

# Train Params
train_param = dict(learning_rate=0.6,
                   learning_rate_min=0.001,
                   momentum=0.9,
                   weight_decay=1e-4,
                   epochs=150,
                   no_wd=True,
                   warm_up_param=dict(warm_up_base_lr=0.02, warm_up_epochs=20))

# Attack Params
attack_param = {'attack': True,
                'epsilon': 8 / 255.,
                'num_steps': 20,
                'step_size': 2 / 255.,
                'random_start': True}
