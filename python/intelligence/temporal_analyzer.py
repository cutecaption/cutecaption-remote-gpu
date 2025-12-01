"""
Temporal Analyzer - EXIF and Time-based Metadata Extraction
Extracts temporal, geographic, and camera metadata from images

Capabilities:
- EXIF data extraction (date, time, camera, settings)
- GPS location extraction
- Time-of-day classification
- Season inference
- Chronological ordering
- Temporal pattern detection
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pytz

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from base_engine import BaseEngine

# Import required libraries
try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    import exifread
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required library: {e}", file=sys.stderr)
    sys.exit(1)


class TemporalAnalyzer(BaseEngine):
    """
    Analyzes temporal and metadata aspects of images
    """
    
    def __init__(self):
        super().__init__("temporal_analyzer", "1.0.0")
        
        # Time classification boundaries
        self.time_periods = {
            'night': (0, 5),
            'dawn': (5, 7),
            'morning': (7, 12),
            'afternoon': (12, 17),
            'evening': (17, 19),
            'dusk': (19, 21),
            'night_late': (21, 24)
        }
        
        # Season boundaries (Northern Hemisphere)
        self.seasons = {
            'winter': [(12, 21), (3, 20)],
            'spring': [(3, 20), (6, 21)],
            'summer': [(6, 21), (9, 22)],
            'autumn': [(9, 22), (12, 21)]
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Temporal Analyzer"""
        try:
            self.logger.info("Initializing Temporal Analyzer...")
            self._mark_initialized()
            
            return {
                "status": "ready",
                "capabilities": [
                    "exif_extraction",
                    "temporal_classification",
                    "gps_extraction",
                    "chronological_ordering"
                ]
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            raise
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process temporal analysis requests"""
        action = request.get('action')
        
        if action == 'health_check':
            return await self.health_check()
        
        elif action == 'extract_temporal':
            return await self._extract_temporal_metadata(
                request['image_path']
            )
        
        elif action == 'batch_extract':
            return await self._batch_extract(
                request['image_paths'],
                request.get('include_gps', True)
            )
        
        elif action == 'find_temporal_patterns':
            return await self._find_temporal_patterns(
                request['temporal_data']
            )
        
        elif action == 'chronological_sort':
            return await self._chronological_sort(
                request['images_with_metadata']
            )
        
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _extract_temporal_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Extract all temporal and camera metadata from an image
        """
        try:
            metadata = {
                "image_path": image_path,
                "temporal": {},
                "camera": {},
                "gps": {},
                "derived": {}
            }
            
            # Try PIL first for standard EXIF
            try:
                img = Image.open(image_path)
                exif_data = img._getexif()
                
                if exif_data:
                    # Extract standard EXIF
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        # Temporal data
                        if tag == 'DateTime':
                            metadata['temporal']['datetime'] = str(value)
                            dt = self._parse_datetime(value)
                            if dt:
                                metadata['temporal']['timestamp'] = dt.timestamp()
                                metadata['temporal']['date'] = dt.date().isoformat()
                                metadata['temporal']['time'] = dt.time().isoformat()
                                metadata['derived']['time_of_day'] = self._classify_time_of_day(dt)
                                metadata['derived']['season'] = self._classify_season(dt)
                                metadata['derived']['day_of_week'] = dt.strftime('%A')
                        
                        elif tag == 'DateTimeOriginal':
                            metadata['temporal']['datetime_original'] = str(value)
                        
                        elif tag == 'DateTimeDigitized':
                            metadata['temporal']['datetime_digitized'] = str(value)
                        
                        # Camera data
                        elif tag in ['Make', 'Model', 'LensMake', 'LensModel']:
                            metadata['camera'][tag.lower()] = str(value)
                        
                        elif tag in ['FocalLength', 'FNumber', 'ExposureTime', 'ISOSpeedRatings']:
                            metadata['camera'][tag.lower()] = value
                        
                        # GPS data
                        elif tag == 'GPSInfo':
                            gps_data = {}
                            for gps_tag in value:
                                sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                                gps_data[sub_tag] = value[gps_tag]
                            
                            # Convert GPS coordinates
                            lat = self._convert_gps_coord(
                                gps_data.get('GPSLatitude'),
                                gps_data.get('GPSLatitudeRef')
                            )
                            lon = self._convert_gps_coord(
                                gps_data.get('GPSLongitude'),
                                gps_data.get('GPSLongitudeRef')
                            )
                            
                            if lat and lon:
                                metadata['gps']['latitude'] = lat
                                metadata['gps']['longitude'] = lon
                                metadata['gps']['coordinates'] = [lat, lon]
                                
                                # Altitude if available
                                if 'GPSAltitude' in gps_data:
                                    metadata['gps']['altitude'] = float(gps_data['GPSAltitude'])
            
            except Exception as pil_error:
                self.logger.debug(f"PIL EXIF extraction failed: {pil_error}")
            
            # Try exifread for more comprehensive extraction
            try:
                with open(image_path, 'rb') as f:
                    tags = exifread.process_file(f, details=False)
                    
                    for tag in tags.keys():
                        if 'DateTime' in tag and not metadata['temporal'].get('datetime'):
                            dt_str = str(tags[tag])
                            dt = self._parse_datetime(dt_str)
                            if dt:
                                metadata['temporal']['datetime'] = dt_str
                                metadata['temporal']['timestamp'] = dt.timestamp()
                                metadata['derived']['time_of_day'] = self._classify_time_of_day(dt)
                                metadata['derived']['season'] = self._classify_season(dt)
            
            except Exception as exif_error:
                self.logger.debug(f"exifread extraction failed: {exif_error}")
            
            # If no EXIF date, use file modification time
            if not metadata['temporal'].get('datetime'):
                file_stat = Path(image_path).stat()
                dt = datetime.fromtimestamp(file_stat.st_mtime)
                metadata['temporal']['datetime'] = dt.isoformat()
                metadata['temporal']['timestamp'] = file_stat.st_mtime
                metadata['temporal']['source'] = 'file_system'
                metadata['derived']['time_of_day'] = self._classify_time_of_day(dt)
                metadata['derived']['season'] = self._classify_season(dt)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract temporal metadata: {e}")
            return {
                "image_path": image_path,
                "error": str(e)
            }
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse various datetime formats"""
        if not dt_str:
            return None
        
        # Common EXIF datetime formats
        formats = [
            '%Y:%m:%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y:%m:%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(str(dt_str).strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def _classify_time_of_day(self, dt: datetime) -> str:
        """Classify time into period of day"""
        hour = dt.hour
        
        for period, (start, end) in self.time_periods.items():
            if start <= hour < end:
                return period.replace('_', ' ')
        
        return 'unknown'
    
    def _classify_season(self, dt: datetime) -> str:
        """Classify date into season (Northern Hemisphere)"""
        month = dt.month
        day = dt.day
        
        for season, boundaries in self.seasons.items():
            start_month, start_day = boundaries[0]
            end_month, end_day = boundaries[1]
            
            if month == start_month and day >= start_day:
                return season
            elif month == end_month and day < end_day:
                return season
            elif start_month < month < end_month:
                return season
        
        return 'unknown'
    
    def _convert_gps_coord(self, coord_parts: Any, ref: str) -> Optional[float]:
        """Convert GPS coordinates from EXIF format to decimal degrees"""
        if not coord_parts or not ref:
            return None
        
        try:
            # Convert degrees, minutes, seconds to decimal
            degrees = float(coord_parts[0])
            minutes = float(coord_parts[1])
            seconds = float(coord_parts[2])
            
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            
            # Apply reference (N/S for latitude, E/W for longitude)
            if ref in ['S', 'W']:
                decimal = -decimal
            
            return round(decimal, 6)
            
        except (IndexError, TypeError, ValueError):
            return None
    
    async def _batch_extract(self, image_paths: List[str], include_gps: bool = True) -> Dict[str, Any]:
        """Extract temporal metadata for multiple images"""
        results = []
        
        for path in image_paths:
            metadata = await self._extract_temporal_metadata(path)
            
            # Optionally exclude GPS for privacy
            if not include_gps:
                metadata.pop('gps', None)
            
            results.append(metadata)
        
        return {
            "count": len(results),
            "images": results
        }
    
    async def _find_temporal_patterns(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in dataset"""
        if not temporal_data:
            return {"patterns": {}}
        
        patterns = {
            "time_of_day_distribution": {},
            "season_distribution": {},
            "day_of_week_distribution": {},
            "hourly_distribution": {},
            "camera_distribution": {},
            "date_range": {},
            "temporal_clusters": []
        }
        
        timestamps = []
        
        for data in temporal_data:
            # Time of day
            tod = data.get('derived', {}).get('time_of_day')
            if tod:
                patterns['time_of_day_distribution'][tod] = \
                    patterns['time_of_day_distribution'].get(tod, 0) + 1
            
            # Season
            season = data.get('derived', {}).get('season')
            if season:
                patterns['season_distribution'][season] = \
                    patterns['season_distribution'].get(season, 0) + 1
            
            # Day of week
            dow = data.get('derived', {}).get('day_of_week')
            if dow:
                patterns['day_of_week_distribution'][dow] = \
                    patterns['day_of_week_distribution'].get(dow, 0) + 1
            
            # Camera
            camera = data.get('camera', {}).get('model')
            if camera:
                patterns['camera_distribution'][camera] = \
                    patterns['camera_distribution'].get(camera, 0) + 1
            
            # Collect timestamps
            ts = data.get('temporal', {}).get('timestamp')
            if ts:
                timestamps.append(ts)
        
        # Date range
        if timestamps:
            patterns['date_range'] = {
                'earliest': datetime.fromtimestamp(min(timestamps)).isoformat(),
                'latest': datetime.fromtimestamp(max(timestamps)).isoformat(),
                'span_days': (max(timestamps) - min(timestamps)) / 86400
            }
            
            # Hourly distribution
            for ts in timestamps:
                hour = datetime.fromtimestamp(ts).hour
                patterns['hourly_distribution'][hour] = \
                    patterns['hourly_distribution'].get(hour, 0) + 1
            
            # Find temporal clusters (burst shooting)
            patterns['temporal_clusters'] = self._find_temporal_clusters(timestamps)
        
        return {"patterns": patterns}
    
    def _find_temporal_clusters(self, timestamps: List[float], threshold_seconds: float = 60) -> List[Dict]:
        """Find clusters of images taken close together in time"""
        if len(timestamps) < 2:
            return []
        
        sorted_ts = sorted(timestamps)
        clusters = []
        current_cluster = [sorted_ts[0]]
        
        for ts in sorted_ts[1:]:
            if ts - current_cluster[-1] <= threshold_seconds:
                current_cluster.append(ts)
            else:
                if len(current_cluster) > 1:
                    clusters.append({
                        'start': datetime.fromtimestamp(current_cluster[0]).isoformat(),
                        'end': datetime.fromtimestamp(current_cluster[-1]).isoformat(),
                        'count': len(current_cluster),
                        'duration_seconds': current_cluster[-1] - current_cluster[0]
                    })
                current_cluster = [ts]
        
        # Add last cluster
        if len(current_cluster) > 1:
            clusters.append({
                'start': datetime.fromtimestamp(current_cluster[0]).isoformat(),
                'end': datetime.fromtimestamp(current_cluster[-1]).isoformat(),
                'count': len(current_cluster),
                'duration_seconds': current_cluster[-1] - current_cluster[0]
            })
        
        return clusters
    
    async def _chronological_sort(self, images_with_metadata: List[Dict]) -> Dict[str, Any]:
        """Sort images chronologically"""
        # Extract timestamp for each image
        items_with_time = []
        
        for item in images_with_metadata:
            timestamp = item.get('temporal', {}).get('timestamp', float('inf'))
            items_with_time.append((timestamp, item))
        
        # Sort by timestamp
        sorted_items = sorted(items_with_time, key=lambda x: x[0])
        
        return {
            "sorted_images": [item for _, item in sorted_items],
            "count": len(sorted_items)
        }


async def main():
    """Main entry point for testing"""
    analyzer = TemporalAnalyzer()
    await analyzer.initialize()
    
    # Run as service
    print(json.dumps({"event": "initialized", "service": "temporal_analyzer"}), flush=True)
    
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = await analyzer.handle_request(request)
            print(json.dumps(response), flush=True)
        except Exception as e:
            error_response = {
                "success": False,
                "error": {"message": str(e)}
            }
            print(json.dumps(error_response), flush=True)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
