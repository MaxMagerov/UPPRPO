FROM gcc:latest as build
WORKDIR /CircularBuffer
COPY . .
RUN apt-get update && apt-get install -y cmake
WORKDIR /CircularBuffer/build
RUN cmake .. && cmake --build . --target Tests

FROM ubuntu:latest
RUN groupadd -r sample && useradd -r -g sample sample
USER sample
WORKDIR /Test
COPY --from=build /CircularBuffer/build/test/Tests .
ENTRYPOINT ["./Tests"]