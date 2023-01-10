from django.contrib import admin
from .models import DiseaseDescription

admin.site.site_header = 'Disease Prediction'
admin.site.index_title = ''
admin.site.site_title = 'disease prediction'


class DiseaseDescriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'disease_name', 'description','image_url')


admin.site.register(DiseaseDescription, DiseaseDescriptionAdmin)