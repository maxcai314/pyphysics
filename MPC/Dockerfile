FROM alpine:latest AS build
RUN apk add build-base cmake
WORKDIR /app
COPY . .
RUN cmake .
RUN make
RUN printf '\x48\xC7\xC0\x01\x00\x00\x00\xC3' | dd conv=notrunc of=MecanumMPC bs=1 seek=$((0x$(nm -a MecanumMPC | grep validate_uid_all_mecanum_mpc | awk -F' ' '{ print $1}'))) # remove licence check
FROM scratch as export-stage
COPY --from=build /app/MecanumMPC .
