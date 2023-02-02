FROM alpine:latest AS build
RUN apk add build-base cmake
WORKDIR /app
COPY . .
RUN cmake .
RUN make

FROM scratch as export-stage
COPY --from=build /app/MecanumMPC .
