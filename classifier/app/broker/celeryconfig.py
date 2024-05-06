from os import getenv


broker_url = getenv("AMQP_URI")
result_backend = getenv("RES_BACKEND_URI")
