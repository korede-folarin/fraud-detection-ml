FROM quay.io/astronomer/astro-runtime:9.6.0

USER astro

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

