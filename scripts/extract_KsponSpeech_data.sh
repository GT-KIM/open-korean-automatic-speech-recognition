find "/app/data/KsponSpeech/KsponSpeech_zip" -name "KsponSpeech_01.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "/app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_01.zip"
find "/app/data/KsponSpeech/KsponSpeech_zip" -name "KsponSpeech_02.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "/app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_02.zip"
find "/app/data/KsponSpeech/KsponSpeech_zip" -name "KsponSpeech_03.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "/app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_03.zip"
find "/app/data/KsponSpeech/KsponSpeech_zip" -name "KsponSpeech_04.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "/app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_04.zip"
find "/app/data/KsponSpeech/KsponSpeech_zip" -name "KsponSpeech_05.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "/app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_05.zip"
find "/app/data/KsponSpeech/KsponSpeech_zip" -name "KsponSpeech_eval.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "/app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_eval.zip"
find "/app/data/KsponSpeech/KsponSpeech_zip" -name "KsponSpeech_scripts.zip.part*" -print0 | sort -zt'.' -k2V | xargs -0 cat > "/app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_scripts.zip"

unzip /app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_01.zip -d /app/data/KsponSpeech/KsponSpeech_raw/
unzip /app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_02.zip -d /app/data/KsponSpeech/KsponSpeech_raw/
unzip /app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_03.zip -d /app/data/KsponSpeech/KsponSpeech_raw/
unzip /app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_04.zip -d /app/data/KsponSpeech/KsponSpeech_raw/
unzip /app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_05.zip -d /app/data/KsponSpeech/KsponSpeech_raw/
unzip /app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_eval.zip -d /app/data/KsponSpeech/KsponSpeech_raw/
unzip /app/data/KsponSpeech/KsponSpeech_zip/KsponSpeech_scripts.zip -d /app/data/KsponSpeech/KsponSpeech_raw/