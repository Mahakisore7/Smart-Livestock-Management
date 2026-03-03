import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

void main() => runApp(const HerdSentinelApp());

class HerdSentinelApp extends StatelessWidget {
  const HerdSentinelApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true, 
        colorSchemeSeed: Colors.blueGrey,
        brightness: Brightness.light,
      ),
      home: const Dashboard(),
    );
  }
}

class Dashboard extends StatefulWidget {
  const Dashboard({super.key});
  @override
  State<Dashboard> createState() => _DashboardState();
}

class _DashboardState extends State<Dashboard> {
  // 🔗 NETWORK CONFIG (Update this IP to match your Pi terminal)
  final String piIP = "10.198.184.233"; 
  
  String selectedCow = ""; 
  List<String> allCows = [];
  Map<String, dynamic> data = {};
  bool connected = false;
  Timer? timer;

  @override
  void initState() {
    super.initState();
    refreshSystem();
    // Auto-refresh every 5 seconds
    timer = Timer.periodic(const Duration(seconds: 5), (t) => refreshSystem());
  }

  @override
  void dispose() {
    timer?.cancel();
    super.dispose();
  }

  // Fetches both the list of cows and the current cow's data
  Future<void> refreshSystem() async {
    await fetchCowList();
    if (selectedCow.isNotEmpty) {
      await fetchData();
    }
  }

  Future<void> fetchCowList() async {
    try {
      final response = await http.get(Uri.parse("http://$piIP:5000/cows"));
      if (response.statusCode == 200) {
        List<String> fetchedCows = List<String>.from(jsonDecode(response.body));
        setState(() {
          allCows = fetchedCows;
          if (selectedCow.isEmpty && allCows.isNotEmpty) {
            selectedCow = allCows.first;
          }
        });
      }
    } catch (e) {
      debugPrint("Discovery Error: $e");
    }
  }

  Future<void> fetchData() async {
    try {
      final response = await http.get(Uri.parse("http://$piIP:5000/latest/$selectedCow"))
          .timeout(const Duration(seconds: 3));
      if (response.statusCode == 200) {
        setState(() {
          data = jsonDecode(response.body);
          connected = true;
        });
      }
    } catch (e) {
      setState(() => connected = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    bool isHealthy = (data['diagnosis'] ?? "Healthy") == "Healthy";
    Color statusColor = isHealthy ? Colors.teal : Colors.redAccent;

    return Scaffold(
      backgroundColor: const Color(0xFFF0F2F5),
      appBar: AppBar(
        title: const Text("🏥 Herd Command", style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: false,
        actions: [
          // 🐄 Multi-Cow Dropdown Selector
          if (allCows.isNotEmpty)
            DropdownButton<String>(
              value: selectedCow,
              underline: const SizedBox(),
              icon: const Icon(Icons.keyboard_arrow_down, color: Colors.blueGrey),
              items: allCows.map((String value) {
                return DropdownMenuItem<String>(
                  value: value,
                  child: Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
                );
              }).toList(),
              onChanged: (newValue) {
                setState(() {
                  selectedCow = newValue!;
                  connected = false;
                });
                fetchData();
              },
            ),
          const SizedBox(width: 15),
        ],
      ),
      body: allCows.isEmpty 
        ? const Center(child: Text("📡 Searching for Cow IDs... Start Sentinel on Pi."))
        : !connected 
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // --- METRICS SECTION ---
                  Row(
                    children: [
                      _metricCard("Body Temp", "${data['temp']}°C", Icons.thermostat, Colors.orange),
                      _metricCard("Coughs", "${data['coughs']}", Icons.graphic_eq, Colors.blue),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      _metricCard("Activity", data['activity'] ?? "Normal", Icons.directions_run, Colors.purple),
                      _metricCard("Status", data['diagnosis'] ?? "Healthy", Icons.psychology, statusColor),
                    ],
                  ),
                  const SizedBox(height: 25),

                  // --- REASONING BOX ---
                  const Text("🤖 VETERINARY AGENT REASONING", 
                    style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blueGrey, fontSize: 12)),
                  const SizedBox(height: 8),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border(left: BorderSide(color: statusColor, width: 8)),
                      boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10)],
                    ),
                    child: Text(
                      data['advice'] ?? "No advice available.",
                      style: const TextStyle(fontSize: 15, height: 1.6, color: Colors.black87),
                    ),
                  ),
                  
                  const SizedBox(height: 25),
                  // --- CHECKLIST ---
                  Container(
                    padding: const EdgeInsets.all(15),
                    decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12)),
                    child: ListTile(
                      leading: Icon(isHealthy ? Icons.check_circle : Icons.warning, color: statusColor),
                      title: Text(isHealthy ? "Routine Monitoring" : "Immediate Isolation Required"),
                      subtitle: Text("Selected: $selectedCow"),
                    ),
                  ),
                ],
              ),
            ),
    );
  }

  Widget _metricCard(String title, String value, IconData icon, Color color) {
    return Expanded(
      child: Card(
        elevation: 0,
        color: Colors.white,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Icon(icon, color: color, size: 24),
              const SizedBox(height: 4),
              Text(title, style: const TextStyle(color: Colors.grey, fontSize: 11)),
              Text(value, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16), overflow: TextOverflow.ellipsis),
            ],
          ),
        ),
      ),
    );
  }
}