model = 'robnet_large_v1'
model_param = dict(C=48,
                   num_classes=1000,
                   layers=20,
                   steps=4,
                   multiplier=4,
                   stem_multiplier=3,
                   share=True,
                   ImgNetBB=True)
dataset = 'ImgNet'
dataset_param = dict(data_root='../data/ImgNet',
                     resize_size=256,
                     input_img_size=224,
                     batch_size=32,
                     num_workers=2)
report_freq = 10
seed = 10
save_path = './log'
resume_path = dict(path='./checkpoint/RobNet_large_v1_imgnet.pth.tar', origin_ckpt=True)

# Attack Params
attack_param = {'attack': True,
                'epsilon': 8 / 255.,
                'num_steps': 20,
                'step_size': 2 / 255.,
                'random_start': True}
