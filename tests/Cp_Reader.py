import tensorflow as tf

reader = tf.train.NewCheckpointReader(
    "/data/rui.wu/Elijha/workspace/Img_emb/work_dir/stack_1_norm_08-19_19-42-03/save-87400")
v = reader.get_variable_to_shape_map()
t = []

for k in v.keys():
    t.append(reader.get_tensor(k))

print()
