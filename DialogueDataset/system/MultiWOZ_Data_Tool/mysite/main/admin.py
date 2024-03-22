from xml.dom.minidom import Document
from django.contrib import admin
from main.models import *
# Register your models here.
# class TestAdmin(admin.ModelAdmin) :
#     list_display = ('chat_id', 'subject_text', 'user_name', 'system_name')
admin.site.register(Room)
admin.site.register(Conversation)
admin.site.register(Scenario)
admin.site.register(Doc)
admin.site.register(Section)
admin.site.register(Span)
