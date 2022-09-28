import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

#######################################

def build_and_compile_model(norm,size_out):
    model = keras.Sequential([
        norm,
        layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(1e-2)),
        layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(1e-2)),
        layers.Dropout(.1, input_shape=(16,)),
        layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(1e-2)),
        layers.Dense(size_out)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.0001))
    return model


def setting_dataframe(df1,df2,scaling=True,popping1='cage_escape_time',otherpop=None):
    if popping1:
        df1.pop(popping1)
    if otherpop:
        for x in list(otherpop):
            df1.pop(x)

    #scaling target dataframe
    disp_scaler= StandardScaler()
    disp_scaler.fit(df2)
    cols=df2.columns
    if scaling:
        df2=pd.DataFrame(disp_scaler.transform(df2),columns=cols)

    #join dataset features e target
    ldf=df1.join(df2)

    train_dataset = ldf.sample(frac=0.8, random_state=2)
    test_dataset = ldf.drop(train_dataset.index)
    train_features = train_dataset.drop(train_dataset.columns[-10:],axis=1)
    test_features = test_dataset.drop(test_dataset.columns[-10:],axis=1)

    train_labels = train_dataset.drop(train_dataset.columns[:-10],axis=1)
    test_labels = test_dataset.drop(test_dataset.columns[:-10],axis=1)

    return train_features, test_features, train_labels, test_labels


########################################
def main():
	gpus = tf.config.list_physical_devices('GPU')

	try:
		tf.config.set_visible_devices(gpus[0], 'GPU')
		tf.config.set_logical_device_configuration(
		        gpus[0],
		        [tf.config.LogicalDeviceConfiguration(memory_limit=38*1024)])

		logical_gpus = tf.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

		temp='41'
		df_041=pd.read_csv('../dataset_0'+temp+'.csv')
		prop_df_041=pd.read_csv('../propensity_various_t_0.'+temp+'.csv')      # propensities

		selected_columns = prop_df_041[['t=1']]
		df_dw=selected_columns.copy()
		df_dw.columns=['isoDW']


		radial_gen_0=pd.read_csv('../boattini_radial_function_gen0_0'+temp+'.csv')
		radial_gen_1=pd.read_csv('../boattini_radial_function_gen1_0'+temp+'.csv')
		radial_gen_2=pd.read_csv('../boattini_radial_function_gen2_0'+temp+'.csv')


		ang_gen0=pd.read_csv('../modified_steinhardt_gen0_0'+temp+'.csv')
		ang_gen1=pd.read_csv('../modified_steinhardt_gen1_0'+temp+'.csv')
		ang_gen2=pd.read_csv('../modified_steinhardt_gen2_0'+temp+'.csv')

		dataframe_list=[df_041,df_dw,
		                radial_gen_0,radial_gen_1,radial_gen_2,
		                ang_gen0,ang_gen1,ang_gen2]

		largedf=pd.concat(dataframe_list,axis=1)

		train_features, test_features, train_labels, test_labels=setting_dataframe(
		    largedf,prop_df_041,
		    otherpop=['v','DW','DW_coarsegraining_1shell',
		              'DW_coarsegraining_2shell',
		       'DW_coarsegraining_3shell'])

		normalizer = tf.keras.layers.Normalization(axis=-1)
		normalizer.adapt(np.array(train_features))


		target_df=prop_df_041
		out_dim=target_df.shape[1]
		dnn_model = build_and_compile_model(normalizer,out_dim)


		my_callbacks = [
		    tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True),
		    tf.keras.callbacks.ModelCheckpoint(filepath='model.h5',save_best_only=True),
		    tf.keras.callbacks.TensorBoard(log_dir='logs'),
		]

		history = dnn_model.fit(
		    train_features,
		    train_labels,
		    batch_size=16,
		    validation_split=0.1,
		    verbose=1, epochs=10000000)

	except RuntimeError as e:
		print(e)



if __name__ == "__main__":
	main()
