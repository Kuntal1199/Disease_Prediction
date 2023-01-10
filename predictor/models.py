from django.db import models


# Create your models here.


class DiseaseDescription(models.Model):
    description = models.TextField(null=True, blank=True)
    image_url = models.URLField(null=True, blank=True)
    disease_name = models.CharField(max_length=500,null=True,blank=True)
    title = models.CharField(max_length=250,null=True,blank=True)
    # def __str__(self):
    #     return self.

