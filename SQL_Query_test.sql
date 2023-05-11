-- delete from Memories;
-- delete from Prompts;
-- delete from Theme_Links;
-- delete from Themes;
-- delete from Memory_States;
-- delete from Memory_Caches;
-- select * from Prompts p where id like '5d49%'

-- select content from Memories where speaker = 'USER' order by created_on desc

-- SELECT speaker, GROUP_CONCAT(content, '~~') AS aggregated_values
-- FROM Memories
-- GROUP BY speaker
-- order by created_on desc;