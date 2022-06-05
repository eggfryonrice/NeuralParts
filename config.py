n_preprocessed_surface_samples = 100000
n_surface_samples = 2000
n_preprocessed_volume_samples = 100000
n_volume_samples = 5000
num_workers = 1
loss_weight = [1.0,0.1,0.01,0.1,0.01]#[1.0,0.0,0.0,0.0,0.0]#[1.0,0.1,0.01,0.1,0.01]
temperature = 0.004
max_shared_pts = 1.95
sphere_radius = 0.25
n_points_coverage = 10
learning_rate = 0.0001
n_epoch = 300
n_step_per_epoch = 500
n_aggregate = 2
n_primitive = 5
batch_size = 4

dfaust_dataset_directory = '/data/D-FAUST'
dfaust_split_file = '/data/dfaust_split.csv'
dfaust_mesh_folder = 'mesh'
dfaust_image_folder = 'image'
dfaust_surface_samples_folder = 'surface_samples'
dfaust_volume_samples_folder = 'volume_samples'


