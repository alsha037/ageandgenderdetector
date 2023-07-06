from django.urls import path
from . import views
urlpatterns = [
    

path('', views.index, name="index"),
# path('display_image', views.display_image, name="display_image"),

]