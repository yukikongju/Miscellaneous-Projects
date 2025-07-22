alter table `relax-melodies-android.sandbox.analytics_events_pc` add column load_timestamp timestamp;
alter table `relax-melodies-android.sandbox.analytics_events_pc` alter column load_timestamp set default current_timestamp();
update `relax-melodies-android.sandbox.analytics_events_pc` set load_timestamp=current_timestamp() where load_timestamp is null;
