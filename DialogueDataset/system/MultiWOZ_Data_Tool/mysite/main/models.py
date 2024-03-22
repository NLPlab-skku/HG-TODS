# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class Conversation(models.Model):
    room_id = models.IntegerField(blank=True, null=True)
    conv_id = models.AutoField(primary_key=True)
    user_name = models.CharField(max_length=100, blank=True, null=True)
    system_name = models.CharField(max_length=100, blank=True, null=True)
    conv_text = models.TextField(blank=True, null=True)
    checked_list = models.TextField(blank=True, null=True)
    flag = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'conversation'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Doc(models.Model):
    doc_id = models.IntegerField(primary_key=True)
    doc_html = models.TextField(blank=True, null=True)
    doc_text = models.TextField(blank=True, null=True)
    title = models.CharField(max_length=100, blank=True, null=True)
    url = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'doc'


class Room(models.Model):
    room_id = models.AutoField(primary_key=True)
    first_date = models.DateField(blank=True, null=True)
    last_date = models.DateField(blank=True, null=True)
    clear = models.IntegerField(blank=True, null=True)
    user_name = models.CharField(max_length=100, blank=True, null=True)
    system_name = models.CharField(max_length=100, blank=True, null=True)
    scenario_id = models.IntegerField(blank=True, null=True)
    errorFlag = models.IntegerField(default=0, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'room'


class Scenario(models.Model):
    scenario_id = models.AutoField(primary_key=True)
    section_list = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'scenario'


class Section(models.Model):
    doc_id = models.IntegerField(blank=True, null=True)
    section_id = models.IntegerField(primary_key=True)
    section_order = models.IntegerField(blank=True, null=True)
    section_text = models.TextField(blank=True, null=True)
    section_html = models.TextField(blank=True, null=True)
    doc_title = models.TextField(blank=True, null=True)
    section_title = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'section'


class Span(models.Model):
    span_id = models.IntegerField(primary_key=True)
    doc_id = models.IntegerField(blank=True, null=True)
    section_id = models.IntegerField(blank=True, null=True)
    span_text = models.TextField(blank=True, null=True)
    span_title = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'span'

