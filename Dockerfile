# CleanAir Routes container for IBM LinuxONE (s390x).
#
# BASE defaults to the official multi-arch Python image. On LinuxONE Community Cloud we build on the
# same image as the Jupyter container, which already has pandas / scikit-learn compiled for s390x:
#   docker build --build-arg BASE=$(docker inspect -f '{{.Config.Image}}' notebook) -t cleanair-routes .
ARG BASE=python:3.11-slim
FROM ${BASE}

USER root
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .

# run as an unprivileged user
RUN (id app >/dev/null 2>&1 || useradd -m app) && chown -R app /app
USER app

ENV HOST=0.0.0.0 PORT=5000
EXPOSE 5000
# reset any start command inherited from the base (e.g. a Jupyter image) so the app runs
ENTRYPOINT []
CMD ["python3", "app.py"]
