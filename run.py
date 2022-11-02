from preprocess import preprocess
from train import train, evaluate

if __name__=="__main__":
	train_path = './nsmc/ratings_train.txt'
	test_path = './nsmc/ratings_test.txt'
	batch_size = 8
	epochs = 2

	train_iter, test_iter, wti, itw, vocab = preprocess(train_path, test_path, batch_size)
	is_train = True
	if is_train:
		loss, acc = train(train_iter, epochs)
	print(loss)
	print(acc)
