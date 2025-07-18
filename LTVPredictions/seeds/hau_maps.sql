create or replace table `relax-melodies-android.mappings.hau_maps` (
    original_hau STRING,
    new_hau STRING
);

insert into `relax-melodies-android.mappings.hau_maps`
values
    ('appstore', 'Organic'),
    ('audiostreaming', 'audacy'),
    ('blog', 'Organic'),
    ('friendfamily', 'Organic'),
    ('healthprofessional', 'Organic'),
    ('influencer', 'partnership'),
    ('no answer', 'no answer'),
    ('other', 'other'),
    ('partners', 'partnership'),
    ('playstore', 'Organic'),
    ('podcast', 'audacy'),
    ('socialmedia', 'Organic'),
    ('socialmedia-facebook', 'Organic'),
    ('socialmedia-instagram', 'Organic'),
    ('socialmedia-snapchat', 'Organic'),
    ('socialmedia-tiktok', 'Organic'),
    ('socialmedia-youtube', 'Organic'),
    ('tiktok', 'tiktokglobal_int'),
    ('tvstreaming', 'tatari_streaming'),
    ('websearch', 'googleadwords_int');
