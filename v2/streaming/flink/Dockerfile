# Flink 1.20 + PyFlink + XGBoost for the v2 streaming scorer (D14)
# PyFlink does not run natively on Windows, so the job runs here.
# Python 3.10 is Ubuntu 22.04's system Python; PyFlink 1.20 supports 3.8-3.11.
# apache-flink 1.20.0 pins apache-beam<2.49, which caps numpy at 1.24 in this
# image; scores still match v1 because the model is loaded from XGBoost's JSON
# format with xgboost 3.2.0 (parity is checked, not assumed — see parity.py).
FROM flink:1.20.0-java17

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir apache-flink==1.20.0 xgboost==3.2.0

# Kafka connector for the DataStream API, matched to Flink 1.20
ADD --chown=flink:flink \
    https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.3.0-1.20/flink-sql-connector-kafka-3.3.0-1.20.jar \
    /opt/flink/lib/flink-sql-connector-kafka-3.3.0-1.20.jar

# checkpoint directory owned by the flink user, so a fresh named volume inherits it
RUN mkdir -p /checkpoints && chown flink:flink /checkpoints
