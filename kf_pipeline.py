from kfp.components import create_component_from_func
from kfp.dsl import pipeline
from preprocess import preprocess
from train import train

prep_op = create_component_from_func(preprocess)
train_op = create_componet_from_func(train)

@pipeline(name="sentiment work flow")
def sentiment_pipeline():
	train_path = './nsmc/ratings_train.txt'
	test_path = './nsmc/ratings_test.txt'
	batch_size = 8
	epochs = 2
	prep = prep_op(train_path, test_path, batch_size)
	result = train_op(prep.outputs['train_iter'], epochs)
	
