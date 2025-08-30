"""
URL configuration for movie_recommendation project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from movies import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('index',views.index),
    path('movies/', views.movie_list, name='movie-list'),
    path('list',views.movie_list),
    path("mov", views.recommend_movies, name="recommend_movies"),
    path("login",views.login),
    path("redirect",views.redirevt),
    path("list-mov",views.list_mov)
]
