namespace API
{
    
    string Setting_BaseURL = "http://127.0.0.1:8080/";
    bool Setting_VerboseLog = false;

	Net::HttpRequest@ Get(const string &in path)
	{
		auto ret = Net::HttpRequest();
		ret.Method = Net::HttpMethod::Get;
		ret.Url = Setting_BaseURL + "api/" + path;
        print("[API] GET " + ret.Url);
		ret.Start();
		return ret;
	}

    Net::HttpRequest@ Post(const string &in path, const string &in body)
    {
        auto ret = Net::HttpPost(Setting_BaseURL + "api/" + path, body, "application/json");
        if (Setting_VerboseLog) {
            print("[API] POST " + ret.Url + " BODY: " + body);
        } else {
            print("[API] POST " + ret.Url);
        }
        return ret;
    }

	Json::Value GetAsync(const string &in path)
	{
		auto req = Get(path);
		while (!req.Finished()) {
			yield();
		}
		return Json::Parse(req.String());
	}
}