import { Suspense, lazy } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppShell } from "@/components/layouts/app-shell";
import { Loader2 } from "lucide-react";

const OverviewPage = lazy(() =>
  import("@/pages/overview-page").then((m) => ({ default: m.OverviewPage })),
);
const LiveDemoPage = lazy(() =>
  import("@/pages/live-demo-page").then((m) => ({ default: m.LiveDemoPage })),
);
const XaiLabPage = lazy(() =>
  import("@/pages/xai-lab-page").then((m) => ({ default: m.XaiLabPage })),
);
const ModelComparePage = lazy(() =>
  import("@/pages/model-compare-page").then((m) => ({ default: m.ModelComparePage })),
);
const ArchitecturePage = lazy(() =>
  import("@/pages/architecture-page").then((m) => ({ default: m.ArchitecturePage })),
);
const ScenesPage = lazy(() =>
  import("@/pages/scenes-page").then((m) => ({ default: m.ScenesPage })),
);
const PresentationPage = lazy(() =>
  import("@/pages/presentation-page").then((m) => ({ default: m.PresentationPage })),
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
  },
});

function PageLoader() {
  return (
    <div className="flex items-center justify-center py-24">
      <Loader2 size={36} className="animate-spin text-cyan-400" />
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<Navigate to="/live" replace />} />
            <Route
              path="/overview"
              element={
                <Suspense fallback={<PageLoader />}>
                  <OverviewPage />
                </Suspense>
              }
            />
            <Route
              path="/live"
              element={
                <Suspense fallback={<PageLoader />}>
                  <LiveDemoPage />
                </Suspense>
              }
            />
            <Route
              path="/xai"
              element={
                <Suspense fallback={<PageLoader />}>
                  <XaiLabPage />
                </Suspense>
              }
            />
            <Route
              path="/compare"
              element={
                <Suspense fallback={<PageLoader />}>
                  <ModelComparePage />
                </Suspense>
              }
            />
            <Route
              path="/arch"
              element={
                <Suspense fallback={<PageLoader />}>
                  <ArchitecturePage />
                </Suspense>
              }
            />
            <Route
              path="/scene"
              element={
                <Suspense fallback={<PageLoader />}>
                  <ScenesPage />
                </Suspense>
              }
            />
            <Route
              path="/present/:section"
              element={
                <Suspense fallback={<PageLoader />}>
                  <PresentationPage />
                </Suspense>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
