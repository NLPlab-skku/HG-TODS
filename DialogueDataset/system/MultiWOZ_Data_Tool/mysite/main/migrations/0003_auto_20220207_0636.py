# Generated by Django 3.2.12 on 2022-02-07 06:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_test_show_flag'),
    ]

    operations = [
        migrations.CreateModel(
            name='AuthGroup',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=150, unique=True)),
            ],
            options={
                'db_table': 'auth_group',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AuthGroupPermissions',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'auth_group_permissions',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AuthPermission',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('codename', models.CharField(max_length=100)),
            ],
            options={
                'db_table': 'auth_permission',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AuthUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128)),
                ('last_login', models.DateTimeField(blank=True, null=True)),
                ('is_superuser', models.IntegerField()),
                ('username', models.CharField(max_length=150, unique=True)),
                ('first_name', models.CharField(max_length=150)),
                ('last_name', models.CharField(max_length=150)),
                ('email', models.CharField(max_length=254)),
                ('is_staff', models.IntegerField()),
                ('is_active', models.IntegerField()),
                ('date_joined', models.DateTimeField()),
            ],
            options={
                'db_table': 'auth_user',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AuthUserGroups',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'auth_user_groups',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AuthUserUserPermissions',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
            ],
            options={
                'db_table': 'auth_user_user_permissions',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Conversation',
            fields=[
                ('room_id', models.IntegerField(blank=True, null=True)),
                ('conv_id', models.AutoField(primary_key=True, serialize=False)),
                ('user_name', models.CharField(blank=True, max_length=100, null=True)),
                ('system_name', models.CharField(blank=True, max_length=100, null=True)),
                ('conv_text', models.TextField(blank=True, null=True)),
                ('checked_list', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'conversation',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoAdminLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('action_time', models.DateTimeField()),
                ('object_id', models.TextField(blank=True, null=True)),
                ('object_repr', models.CharField(max_length=200)),
                ('action_flag', models.PositiveSmallIntegerField()),
                ('change_message', models.TextField()),
            ],
            options={
                'db_table': 'django_admin_log',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoContentType',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('app_label', models.CharField(max_length=100)),
                ('model', models.CharField(max_length=100)),
            ],
            options={
                'db_table': 'django_content_type',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoMigrations',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('app', models.CharField(max_length=255)),
                ('name', models.CharField(max_length=255)),
                ('applied', models.DateTimeField()),
            ],
            options={
                'db_table': 'django_migrations',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='DjangoSession',
            fields=[
                ('session_key', models.CharField(max_length=40, primary_key=True, serialize=False)),
                ('session_data', models.TextField()),
                ('expire_date', models.DateTimeField()),
            ],
            options={
                'db_table': 'django_session',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Doc',
            fields=[
                ('doc_id', models.IntegerField(primary_key=True, serialize=False)),
                ('doc_html', models.TextField(blank=True, null=True)),
                ('doc_text', models.TextField(blank=True, null=True)),
                ('title', models.CharField(blank=True, max_length=100, null=True)),
                ('url', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'doc',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='MainTest',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('chat_id', models.CharField(max_length=10)),
                ('subject_text', models.CharField(max_length=100)),
                ('user_name', models.CharField(max_length=10)),
                ('system_name', models.CharField(max_length=10)),
                ('show_flag', models.IntegerField()),
            ],
            options={
                'db_table': 'main_test',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Room',
            fields=[
                ('room_id', models.AutoField(primary_key=True, serialize=False)),
                ('first_date', models.DateField(blank=True, null=True)),
                ('last_date', models.DateField(blank=True, null=True)),
                ('clear', models.IntegerField(blank=True, null=True)),
                ('user_name', models.CharField(blank=True, max_length=100, null=True)),
                ('system_name', models.CharField(blank=True, max_length=100, null=True)),
                ('scenario_id', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'room',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Scenario',
            fields=[
                ('scenario_id', models.AutoField(primary_key=True, serialize=False)),
                ('section_list', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'scenario',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Section',
            fields=[
                ('doc_id', models.IntegerField(blank=True, null=True)),
                ('section_id', models.IntegerField(primary_key=True, serialize=False)),
                ('section_order', models.IntegerField(blank=True, null=True)),
                ('section_text', models.TextField(blank=True, null=True)),
                ('section_html', models.TextField(blank=True, null=True)),
                ('title', models.CharField(blank=True, max_length=100, null=True)),
            ],
            options={
                'db_table': 'section',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Span',
            fields=[
                ('span_id', models.IntegerField(primary_key=True, serialize=False)),
                ('doc_id', models.IntegerField(blank=True, null=True)),
                ('section_id', models.IntegerField(blank=True, null=True)),
                ('span_text', models.TextField(blank=True, null=True)),
                ('span_title', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'span',
                'managed': False,
            },
        ),
        migrations.DeleteModel(
            name='Test',
        ),
    ]