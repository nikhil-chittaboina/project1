from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class predicting_behavior_change(models.Model):

    Fid= models.CharField(max_length=300)
    Certification_Course= models.CharField(max_length=300)
    Gender= models.CharField(max_length=300)
    Department= models.CharField(max_length=300)
    Height_CM= models.CharField(max_length=300)
    Weight_KG= models.CharField(max_length=300)
    Tenth_Mark= models.CharField(max_length=300)
    Twelth_Mark= models.CharField(max_length=300)
    hobbies= models.CharField(max_length=300)
    daily_studing_time= models.CharField(max_length=300)
    prefer_to_study_in= models.CharField(max_length=300)
    like_your_degree= models.CharField(max_length=300)
    social_medai_video= models.CharField(max_length=300)
    Travelling_Time= models.CharField(max_length=300)
    Stress_Level= models.CharField(max_length=300)
    Financial_Status= models.CharField(max_length=300)
    alcohol_consumption= models.CharField(max_length=300)
    part_time_job= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



