FROM alpine:latest AS build
RUN apk add build-base cmake pcre-tools gcompat
WORKDIR /app
COPY . .
RUN cmake .
RUN make
RUN printf '\x48\xC7\xC0\x01\x00\x00\x00\xC3' | dd conv=notrunc of=MecanumMPC bs=1 seek=$(($(objdump -F -D MecanumMPC | pcregrep -o '(?<=validate_uid_all_mecanum_mpc> \(File Offset: )0x[a-f\d]+(?=\):)'))) # remove licence check
FROM scratch as export-stage
COPY --from=build /app/MecanumMPC .
