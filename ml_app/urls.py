from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('graphs/', views.graphs, name='graphs'),
    path('predict/', views.predict, name='predict'),
    path('predict/upload/', views.predict_upload, name='predict_upload'),
    path('train/', views.train_page, name='train'),
    path('train/start/', views.start_training, name='start_training'),
    path('train/progress/', views.get_training_progress, name='training_progress'),
]

