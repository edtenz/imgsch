import os

############### Number of log files ###############
LOGS_NUM = int(os.getenv("logs_num", "0"))

############### Detectserver Configuration ###############
PORT = os.getenv("PORT", "8099")
