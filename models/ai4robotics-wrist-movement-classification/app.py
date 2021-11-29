import grpc
from concurrent import futures
import time
import wrist_classifier_pb2
import wrist_classifier_pb2_grpc
import numpy as np
from joblib import load


port = 8061

class WristMovementClassifier(wrist_classifier_pb2_grpc.WristMovementClassifierServicer):

    def __init__(self):
        self.clf = load("wristdata1024_raw_svm_classifier.joblib")
        
    # def startTraining(self, request, context):
    #     print("start training")
    #     x_train = np.load(request.training_data_filename)
    #     y_labels = np.load(request.training_labels_filename)
    #     split_border = int(len(x_train) * request.validation_ratio)
    #     x_val = x_train[:split_border]
    #     partial_x_train = x_train[split_border:]
    #     y_val = y_labels[:split_border]
    #     partial_y_train = y_labels[split_border:]

    #     history = self.model.fit(partial_x_train, partial_y_train, request.epochs,
    #                              batch_size=request.batch_size, validation_data=(x_val, y_val), callbacks=[tensorboard_callback])
    #     self.model.save(request.model_path)

    #     response = news_classifier_pb2.TrainingStatus()
    #     response.accuracy = history.history['accuracy'][-1]
    #     response.validation_loss = history.history['val_loss'][-1]
    #     response.status_text = 'success'
    #     return response

    def classify(self, request, context):
        response = wrist_classifier_pb2.Movement()
        prediction = self.clf.predict(np.array([request.sensor_data]))
        response.movement = prediction
        return response

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
wrist_classifier_pb2_grpc.add_WristMovementClassifierServicer_to_server(WristMovementClassifier(), server)
print("Starting server. Listening on port : " + str(port))
server.add_insecure_port("[::]:{}".format(port))
server.start()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)