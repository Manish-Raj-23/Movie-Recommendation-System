from django.db import models

# Create your models here.
from django.db import models

class Movie(models.Model):
    title = models.CharField(max_length=255)
    poster = models.URLField()
    description = models.TextField()
    rating = models.FloatField()
    director = models.CharField(max_length=255)

    def __str__(self):
        return self.title
