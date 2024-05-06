from celery import Celery

import broker.celeryconfig as celeryconfig


celery_app = Celery("worker")
celery_app.config_from_object(celeryconfig)
