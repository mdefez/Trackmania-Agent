uint nextSend = 0;
uint SEND_EVERY_MS = 1000;

void Main() {
    while (true) {
        if (Time::Now >= nextSend) {
            nextSend = Time::Now + SEND_EVERY_MS;
            print(Json::Write(DumpData(), false));
            API::Post("data", Json::Write(DumpData(), false));
        }
        yield();
    }
}


void OnDestroyed() {
}