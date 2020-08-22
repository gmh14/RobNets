model = 'robnet_free'
model_param = dict(C=36,
                   num_classes=10,
                   layers=20,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=False,
                   AdPoolSize=1)
dataset = 'SVHN'
dataset_param = dict(data_root='../data/SVHN',
                     batch_size=32,
                     num_workers=2)
report_freq = 10
seed = 10

save_path = './log'
resume_path = dict(path='./checkpoint/RobNet_free_SVHN.pth.tar', origin_ckpt=True)

# Attack Params
attack_param = {'attack': True,
                'epsilon': 8 / 255.,
                'num_steps': 100,
                'step_size': 2 / 255.,
                'random_start': True}
