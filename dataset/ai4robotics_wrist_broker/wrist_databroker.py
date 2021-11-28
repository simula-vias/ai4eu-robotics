import csv
import gzip

from concurrent import futures
import time
import grpc
import wrist_databroker_pb2_grpc
import wrist_databroker_pb2

port = 8061


class WristSensorDatabroker(wrist_databroker_pb2_grpc.WristSensorDatabrokerServicer):
    def __init__(self):
        self.reader = self._make_reader("datafile.csv.gz")

    def _make_reader(self, fname):
        with gzip.open(fname, "rt") as f:
            csvreader = csv.DictReader(f)
            for line in csvreader:
                yield line

    def get_next(self, request, context):
        response = wrist_databroker_pb2.Measurement()

        try:
            row = self.reader.next()
            response.index = row["index"]
            response.phase = row["phase"]
            response.pattern = row["pattern"]
            response.iteration = row["iteration"]

            sensor_data = []
            for k, v in row.items():
                if not k.startswith("s1"):
                    continue

                sensor_data.append(float(v))

            response.sensor_data = sensor_data
        except StopIteration:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("all data has been processed")

        return response


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
wrist_databroker_pb2_grpc.add_WristSensorDatabrokerServicer_to_server(WristSensorDatabroker(), server)
print("Starting server. Listening on port : " + str(port))
server.add_insecure_port("[::]:{}".format(port))
server.start()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
