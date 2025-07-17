create or replace table `relax-melodies-android.mappings.hau_maps` (
    original_hau STRING,
    new_hau STRING
);

insert into `relax-melodies-android.mappings.hau_maps`
values
    ('appstore', ''), -- verify
    ('audiostreaming', 'audacy'),
    ('blog', 'Organic'),
    ('friendfamily', 'Organic'),
    ('healthprofessional', 'Organic'),
    ('influencer', ''), -- verify
    ('no answer', 'no answer'),
    ('other', 'other'),
    ('partners', ''), -- verify
    ('playstore', ''), -- verify
    ('podcast', 'audacy'),
    ('socialmedia', ''), -- verify
    ('socialmedia-facebook', ''),
    ('socialmedia-instagram', ''),
    ('socialmedia-snapchat', ''),
    ('socialmedia-tiktok', ''),
    ('socialmedia-youtube', ''),
    ('tiktok', 'tiktokglobal_int'),
    ('tvstreaming', 'tatari_streaming'),
    ('websearch', 'googleadwords_int');
