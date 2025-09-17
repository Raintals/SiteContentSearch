from django.urls import path
from .views import SearchAPIView

urlpatterns = [
    path("api/search/", SearchAPIView.as_view(), name="search")
]
