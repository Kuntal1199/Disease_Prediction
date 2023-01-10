from django.urls import path
from . import views

urlpatterns = [
    path('checkdiabetes', views.checkdiabetes, name='checkdiabetes'),
    path('checkbreastcancer', views.checkbreastcancer, name='checkbreastcancer'),
    path('checklungcancer', views.checklungcancer, name='checklungcancer'),
    path('checkkidneydisease', views.checkkidneydisease, name='checkkidneydisease'),
    path('brain_tumor', views.brain_tumor, name="brain_tumor"),
    path('heartdisease', views.heartdisease, name="heartdisease"),
    path('front', views.front, name='front'),
    path('home', views.home, name='home'),
    path('form/<int:id>', views.disease_form, name='disease_detail'),
    path('terms_condition', views.terms_condition, name='terms_condition')
]
