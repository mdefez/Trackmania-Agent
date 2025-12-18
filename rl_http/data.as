Json::Value DumpVehicleData() {
    Json::Value data = Json::Object();

    uint t;
    float spd;
    float sspd;
    vec3 pos;
    float steer;
    float throttle;
    float brake;
    bool finished;
    
    bool groundContact;

    // Front Left Wheel Data
    float fl_steer;
    float fl_slip;

    // Front Right Wheel Data
    float fr_steer;
    float fr_slip;

    // Rear Left Wheel Data
    float rl_slip;

    // Rear Right Wheel Data
    float rr_slip;

    CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
    if (vis is null) {
        print("null");
        return data;
    } else {
        t          = Time::Now;
        pos        = vis.Position;
        spd        = vis.WorldVel.Length();
#if SIG_SCHOOL
        sspd       = VehicleState::GetSideSpeed(vis);
#else
        sspd       = 0.0f;
#endif
        steer      = vis.InputSteer;
        throttle   = vis.InputGasPedal;
        brake      = vis.InputBrakePedal;
        groundContact = vis.IsGroundContact;

        // Direction
        direction = vis.Dir;

        // Wheel data
        fl_steer = vis.FLSteerAngle;
        fl_slip  = vis.FLSlipCoef;

        fr_steer = vis.FRSteerAngle;
        fr_slip  = vis.FRSlipCoef;

        rl_slip  = vis.RLSlipCoef;
        rr_slip  = vis.RRSlipCoef;

        // Construct JSON
        data["time"] = t;

        // Position
        auto position = Json::Array();
        position.Add(pos.x);
        position.Add(pos.y);
        position.Add(pos.z);
        data["position"] = position;

        // Direction
        auto dir = Json::Array();
        dir.Add(direction.x);
        dir.Add(direction.y);
        dir.Add(direction.z);
        data["direction"] = dir;

        data["speed"] = spd;
        data["sideSpeed"] = sspd;
        data["steer"] = steer;
        data["throttle"] = throttle;
        data["brake"] = brake;
        data["finished"] = finished;
        data["groundContact"] = groundContact;

        Json::Value wheels = Json::Object();
        Json::Value fl = Json::Object();
        fl["steer"] = fl_steer;
        fl["slip"]  = fl_slip;

        Json::Value fr = Json::Object();
        fr["steer"] = fr_steer;
        fr["slip"]  = fr_slip;

        Json::Value rl = Json::Object();
        rl["slip"]  = rl_slip;

        Json::Value rr = Json::Object();
        rr["slip"]  = rr_slip;

        wheels["FL"] = fl; wheels["FR"] = fr; wheels["RL"] = rl; wheels["RR"] = rr;
        data["wheels"] = wheels;

        return data;
    }
}



// General data dump
Json::Value DumpData() {
    auto rd = MLFeed::GetRaceData_V4();

    Json::Value root = Json::Object();

    Json::Value metadata = Json::Object();

    // Map
    metadata["map"] = rd.Map;

    // Spawn count
    metadata["spawnCount"] = rd.SpawnCounter;

    // Checkpoints
    metadata["cpCount"] = rd.CPCount;

    root["metadata"] = metadata;

    auto vehicleData = DumpVehicleData();
    root["vehicleData"] = vehicleData;

    return root;
}