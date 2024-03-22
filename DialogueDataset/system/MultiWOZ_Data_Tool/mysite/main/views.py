from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Room
from .models import Scenario
from .models import Section
from .models import Span
from .models import Conversation
import json
from django.contrib import messages
from django.core import serializers
# from .script.main import util

# Create your views here.
def mainpage(request):
    return render(request, 'main/mainpage.html')

def chatlistpage(request):
    chat_list = Room.objects.filter(clear = 0, errorFlag=0)
    context = {
        'chat_list':chat_list,
    }
    return render(request, 'main/chatlist.html', context)

def chatroompage(request, room_id, role):
    room = Room.objects.get(room_id = room_id)
    if request.method == "POST":
        last_conv = Conversation.objects.filter(room_id = room_id).order_by('-conv_id')
        if request.POST['chat'] and request.POST.getlist('spans') :
            if len(last_conv) > 1 and  ((role == 'System' and last_conv[0].flag == 0 and last_conv[1].flag == 0) or (role == 'User' and last_conv[0].flag == 1 and last_conv[1].flag == 1)):
                messages.warning(request, '3번이상 연속된 발화는 허용되지 않습니다.')
            else :
                if len(last_conv) > 0 and last_conv[0].conv_text == request.POST['chat'] and last_conv[0].checked_list == ' '.join(request.POST.getlist('spans')):
                    return redirect('/chatroom/'+str(room_id)+'/'+str(role))
                if '0' in request.POST.getlist('spans') and len(request.POST.getlist('spans')) > 1:
                    messages.warning(request, "'해당사함 없음'은 다른 문장과 함께 선택될 수 없습니다.")
                else:
                    if role == 'System' :
                        flag = 0
                    else :
                        flag = 1
                    Conversation.objects.create(
                        room_id = room_id,
                        user_name = room.user_name,
                        system_name = room.system_name,
                        conv_text = request.POST['chat'],
                        checked_list = ' '.join(request.POST.getlist('spans')),
                        flag = flag
                    )
        else :
            messages.warning(request, '발화를 입력하거나 발화에 참고한 문장을 체크해주세요.')
    
    if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' and request.method == "GET":
        try:
            span_list = Span.objects.filter(section_id = request.GET['section_id'])
            section = Section.objects.get(section_id = request.GET['section_id'])
            response = {
                'span_list': list(span_list.values()),
                'doc_title' : section.doc_title,
                'section_title' : section.section_title,
                'section_html' : section.section_html
            }
            return JsonResponse(response)
        except:
            print('error: ', request.GET)

    scenario = Scenario.objects.get(scenario_id = room.scenario_id)
    section_list = list(map(
        lambda x: Section.objects.get(section_id = int(x)),
        scenario.section_list.split(' ')
    ))
    span_list = Span.objects.filter(section_id = section_list[0].section_id)
    conversation_list = Conversation.objects.filter(room_id = room_id)
    context = {
        'role': role,
        'section_list': section_list,
        'span_list': span_list,
        'conversation_list' : conversation_list,
        'room_id' : room_id,
    }
    return render(request, 'main/chatroom.html', context)

def clearRoom(request) :
    jsonObject = json.loads(request.body)
    room = Room.objects.get(room_id = jsonObject.get('room_id'))
    room.clear = 1
    room.save()
    return JsonResponse(jsonObject)

def deleteRoom(request) :
    jsonObject = json.loads(request.body)
    room = Room.objects.get(room_id = jsonObject.get('room_id'))
    room.errorFlag = 1
    room.save()
    return JsonResponse(jsonObject)

def checkDB(request) :
    receive_message = request.POST.get('send_data')
    conversation_list = Conversation.objects.filter(room_id = receive_message)
    send_message = {'send_data' : serializers.serialize("json", conversation_list)}
    return JsonResponse(send_message)

def selectpage(request, room_id):
    return render(request, 'main/select.html', {'room_id':room_id})

def makepage(request):
    if request.method == "POST":
        newRoom = Room.objects.create(
            user_name = request.POST['user_name'],
            system_name = request.POST['system_name'],
        )
        newRoom.scenario_id = newRoom.room_id
        newRoom.clear = 0
        newRoom.save()
        return chatlistpage(request)
    return render(request, 'main/make.html')
