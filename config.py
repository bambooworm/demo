# Freesound Dataset Kaggle 2018
# Application configurations

from easydict import EasyDict

conf = EasyDict()

# Basic configurations
conf.sampling_rate = 16000
conf.duration = 1
conf.hop_length = 253 # to make time steps 64
conf.fmin = 20
conf.fmax = conf.sampling_rate // 2
conf.n_mels = 64
conf.n_fft = conf.n_mels * 20
conf.model = 'alexnet' #'mobilenetv2' #
# conf.sampling_rate = 44100
# conf.duration = 1
# conf.hop_length = 197 # to make time steps 224
# conf.fmin = 20
# conf.fmax = conf.sampling_rate // 2
# conf.n_mels = 224
# conf.n_fft = conf.n_mels * 20
# conf.model = 'mobilenetv2' #'alexnet' #

# Labels
# conf.labels = [ 'Accelerating_and_revving_and_vroom', 'Accordion', 'Acoustic_guitar', 'Applause', 'Bark',
#                 'Bass_drum', 'Bass_guitar', 'Bathtub_(filling_or_washing)', 'Bicycle_bell',
#                 'Burping_and_eructation', 'Bus', 'Buzz', 'Car_passing_by', 'Cheering',
#                 'Chewing_and_mastication', 'Child_speech_and_kid_speaking', 'Chink_and_clink',
#                 'Chirp_and_tweet', 'Church_bell', 'Clapping', 'Computer_keyboard', 'Crackle', 'Cricket',
#                 'Crowd', 'Cupboard_open_or_close', 'Cutlery_and_silverware', 'Dishes_and_pots_and_pans',
#                 'Drawer_open_or_close', 'Drip', 'Electric_guitar', 'Fart', 'Female_singing',
#                 'Female_speech_and_woman_speaking', 'Fill_(with_liquid)', 'Finger_snapping',
#                 'Frying_(food)', 'Gasp', 'Glockenspiel', 'Gong', 'Gurgling', 'Harmonica', 'Hi-hat', 'Hiss',
#                 'Keys_jangling', 'Knock', 'Male_singing', 'Male_speech_and_man_speaking',
#                 'Marimba_and_xylophone', 'Mechanical_fan', 'Meow', 'Microwave_oven', 'Motorcycle',
#                 'Printer', 'Purr', 'Race_car_and_auto_racing', 'Raindrop', 'Run', 'Scissors', 'Screaming',
#                 'Shatter', 'Sigh', 'Sink_(filling_or_washing)', 'Skateboard', 'Slam', 'Sneeze', 'Squeak',
#                 'Stream', 'Strum', 'Tap', 'Tick-tock', 'Toilet_flush', 'Traffic_noise_and_roadway_noise',
#                 'Trickle_and_dribble', 'Walk_and_footsteps', 'Water_tap_and_faucet', 'Waves_and_surf',
#                 'Whispering', 'Writing', 'Yell', 'Zipper_(clothing)' ]
conf.labels = [ 'car', 'none' ]

# Training Configurations
conf.folder = '.'
conf.n_fold = 1
conf.normalize = 'samplewise'
conf.valid_limit = None
conf.random_state = 42
conf.test_size = 0.01
conf.samples_per_file = -1
conf.batch_size = 32
conf.learning_rate = 0.000002
conf.epochs = 500
conf.verbose = 1
conf.best_weight_file = 'best_alexnet_weight.h5'

# Runtime conficurations
conf.rt_process_count = 1
conf.rt_oversamples = 10
conf.pred_ensembles = 10
conf.runtime_model_file = 'best_alexnet_weight.pb' ### NOT PROVIDED
